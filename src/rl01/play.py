"""回放 / 评估一个检查点。

用法：
    uv run python src/rl01/play.py runs/<名字>/model_500.pt --vx 1.0          # 打开查看器，一直开着，关窗口退出
    uv run python src/rl01/play.py runs/<名字>/model_500.pt --sweep           # 查看器里每 5 秒换一个指令速度 0.3/0.5/0.8/1/1.5/2
    uv run python src/rl01/play.py runs/<名字>/model_500.pt --eval            # 无界面：各速度测 64 个环境 × 10 秒，有随机推力
    uv run python src/rl01/play.py "低速.pt,高速.pt" --sweep                    # 混合策略：<0.8 用低速、>=1.0 用高速，中间线性过渡；可加 ,v_lo,v_hi
    --push 0 关闭推力；--vx 指令速度（m/s）
"""
import argparse
import time

import mujoco
import numpy as np
import torch

from rl01.env import N_ACT, N_CRITIC_EXTRA, N_OBS, EnvCfg, Go2Env, build_train_model
from rl01.blend import BlendedPolicy
from rl01.ppo import ActorCritic


def load_policy(path):
    """path 可以是单个检查点，也可以是 "低速.pt,高速.pt[,v_lo,v_hi]"：按速度指令混合两个策略（见 blend.py）。"""
    if "," in path:
        parts = path.split(",")
        low, _ = load_policy(parts[0])
        high, ck = load_policy(parts[1])
        v_lo, v_hi = (float(parts[2]), float(parts[3])) if len(parts) == 4 else (0.8, 1.0)
        return BlendedPolicy(low, high, v_lo, v_hi), ck
    ck = torch.load(path, map_location="cpu")
    if "blend" in ck:  # 打包好的混合检查点（blend.py 生成）
        b = ck["blend"]
        pols = []
        for key in ("low_model", "high_model"):
            m = ActorCritic(N_OBS, N_OBS + N_CRITIC_EXTRA, N_ACT)
            m.load_state_dict(b[key])
            m.eval()
            pols.append(m)
        return BlendedPolicy(pols[0], pols[1], b["v_lo"], b["v_hi"]), ck
    actor_dim = ck["model"]["actor.0.weight"].shape[1]
    critic_dim = ck["model"]["critic.0.weight"].shape[1]
    ac = ActorCritic(actor_dim, critic_dim, N_ACT)
    ac.load_state_dict(ck["model"])
    ac.eval()
    return ac, ck


def make_env(ck, n, push, seed=123):
    d = dict(ck["env_cfg"])
    for k in ("friction_range", "mass_range", "obs_taps", "kp_range", "kd_range", "damping_range"):
        if k in d:
            d[k] = tuple(d[k])
    d["stairs"] = False
    d["flat_frac"] = 0.0
    d["episode_s"] = 20.0
    cfg = EnvCfg(**d)
    cfg.push_vel = push
    if push <= 0:
        cfg.push_ang = 0.0
        cfg.push_z = 0.0
    return Go2Env(n, cfg, seed=seed, noise=False)


def set_cmd(env, vx):
    env.cmd[:] = [vx, 0.0, 0.0]
    env.cmd_left[:] = 10**9
    env._compute_obs(shift=False)


@torch.no_grad()
def policy_action(ac, env):
    obs, _ = env.get_obs()
    return ac.actor(torch.as_tensor(obs)).numpy()


def evaluate(ac, ck, push):
    print(f"检查点迭代 {ck['iteration']}，训练时的指令上限 {ck['vmax']:.2f} m/s，推力 {push} m/s")
    print(f"{'指令 vx':>8} {'实际 vx':>8} {'误差':>7} {'跟踪(0-1)':>10} {'摔倒率':>7}")
    for vx in [0.0, 0.5, 1.0, 1.5, 2.0]:
        env = make_env(ck, 64, push)
        set_cmd(env, vx)
        vs = []
        for k in range(500):  # 10 秒
            env.step(policy_action(ac, env))
            set_cmd(env, vx)
            if k >= 100:
                vs.append(env.lin_b[:, 0].mean())
        s = env.pop_stats()
        err = abs(np.mean(vs) - vx)
        print(f"{vx:8.2f} {np.mean(vs):8.2f} {err:7.2f} {s['track_lin'] / max(s['steps'], 1):10.3f} {s['falls'] / 64:7.2f}")


def view(ac, ck, vx, sweep, push):
    import mujoco.viewer

    env = make_env(ck, 1, push)
    model = build_train_model(visual=True)
    data = mujoco.MjData(model)
    cmds = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]  # 覆盖低速和高速两段
    t_start, last_print = time.time(), 0.0
    with mujoco.viewer.launch_passive(model, data) as v:
        step = 0
        while v.is_running():
            t0 = time.perf_counter()
            cmd = cmds[int(step * env.dt / 5) % len(cmds)] if sweep else vx
            set_cmd(env, cmd)
            env.step(policy_action(ac, env))
            mujoco.mj_setState(model, data, env.state[0], mujoco.mjtState.mjSTATE_FULLPHYSICS)
            mujoco.mj_forward(model, data)
            v.cam.lookat[:] = data.qpos[:3]
            v.sync()
            step += 1
            if time.time() - last_print > 2:
                last_print = time.time()
                print(f"指令 vx={cmd:.2f}  实际 vx={env.lin_b[0, 0]:+.2f}  机身高度={env.state[0, 3]:.3f}")
            rest = env.dt - (time.perf_counter() - t0)
            if rest > 0:
                time.sleep(rest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--vx", type=float, default=1.0)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--push", type=float, default=0.6)
    args = ap.parse_args()
    torch.set_num_threads(2)
    ac, ck = load_policy(args.checkpoint)
    if args.eval:
        evaluate(ac, ck, args.push)
    else:
        view(ac, ck, args.vx, args.sweep, args.push)


if __name__ == "__main__":
    main()
