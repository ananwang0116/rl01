"""以真实时间回放楼梯策略，关掉 MuJoCo 窗口结束。

例如：uv run python src/rl01/stairs_play.py runs/go2_stairs_v4/model_100.pt --height 5 --steps 5
"""
import argparse
import time

import mujoco
import numpy as np
import torch

from rl01.env import EnvCfg, Go2Env, STAIR_STAGES, build_train_model
from rl01.play import load_policy


def main():
    parser = argparse.ArgumentParser(description='查看 Go2 楼梯策略（MuJoCo 仿真）')
    parser.add_argument('checkpoint', help='训练保存的 .pt 检查点')
    parser.add_argument('--height', type=int, choices=[5, 10, 15], default=5, help='每级高度，单位 cm')
    parser.add_argument('--steps', type=int, choices=[5, 10, 15, 20], default=5, help='台阶级数')
    parser.add_argument('--vx', type=float, default=0.33, help='前进指令 m/s，楼梯上限 1.0 m/s')
    parser.add_argument('--push', type=float, default=0.0, help='随机推力速度上限 m/s，默认关闭')
    parser.add_argument('--seed', type=int, default=719)
    parser.add_argument('--headless-seconds', type=float, default=0, help='不打开窗口，运行指定秒数检查回放')
    args = parser.parse_args()
    torch.set_num_threads(2)
    policy, ck = load_policy(args.checkpoint)
    d = dict(ck['env_cfg'])
    for k in ('friction_range', 'mass_range', 'obs_taps', 'kp_range', 'kd_range', 'damping_range'):
        if k in d:
            d[k] = tuple(d[k])
    d['flat_frac'] = 0.0
    d['stair_mix_prev'] = 0.0
    d['stair_all_stages'] = False
    d['stair_speed_curriculum'] = False
    d['cmd_slew_rate'] = 0.0
    cfg = EnvCfg(**d)
    cfg.stairs = True
    cfg.noise = False
    cfg.push_vel = args.push
    if args.push <= 0:
        cfg.push_ang = 0.0
        cfg.push_z = 0.0
    cfg.episode_s = 45.0
    env = Go2Env(1, cfg, seed=args.seed, nthread=1)
    stage = STAIR_STAGES.index((args.height / 100, args.steps))
    env.stage = stage
    env.reset_idx(np.arange(1))
    env.cmd[:] = [args.vx, 0, 0]
    env.cmd_left[:] = 10**9
    env._compute_obs(shift=False)
    print(f'检查点 {args.checkpoint}；楼梯高度 {args.height} cm / 踏面 25 cm / {args.steps} 级；指令 {args.vx:.2f} m/s；推力 {args.push:.2f}', flush=True)

    @torch.no_grad()
    def advance():
        obs, _ = env.get_obs()
        action = policy.actor(torch.as_tensor(obs)).numpy()
        _, done, _ = env.step(action)
        if done[0]:
            success, fall, timeout = env.last_outcome
            why = '到达平台' if success[0] else '摔倒或偏离' if fall[0] else '超时'
            print(f'{why}，环境已重置', flush=True)
        env.cmd[:] = [args.vx, 0, 0]
        env.cmd_left[:] = 10**9
        env._compute_obs(shift=False)

    if args.headless_seconds > 0:
        for _ in range(round(args.headless_seconds / env.dt)):
            advance()
        print(f'无界面检查完成：有限状态={bool(np.isfinite(env.state).all())}', flush=True)
        return

    import mujoco.viewer
    model = build_train_model(visual=True, stair_stage=stage, stair_top=cfg.stair_top)
    # Directional-light shadow mapping creates moving black blocks on this scene.
    # This is the display model only; the physics models remain unchanged.
    model.light_castshadow[:] = False
    data = mujoco.MjData(model)
    print('回放窗口已打开；关闭窗口结束。', flush=True)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -12
        viewer.cam.distance = 3.0
        while viewer.is_running():
            t0 = time.perf_counter()
            advance()
            mujoco.mj_setState(model, data, env.state[0], mujoco.mjtState.mjSTATE_FULLPHYSICS)
            mujoco.mj_forward(model, data)
            viewer.cam.lookat[:] = [env.state[0, 1] + 0.45, env.state[0, 2], env.state[0, 3]]
            viewer.sync()
            wait = env.dt - (time.perf_counter() - t0)
            if wait > 0:
                time.sleep(wait)


if __name__ == '__main__':
    main()
