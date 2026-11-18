"""训练 Go2 速度指令策略（0 到 2 m/s，带随机推力、观测噪声、摩擦/质量随机化）。

一"步" = 一次 PPO 迭代（采 steps × num_envs 个样本，然后更新网络）。
- 终端输出：每 2.5% 一次（总数 20000 时每 500 步）
- 保存检查点：每 10% 一次（总数 20000 时每 2000 步），文件在 runs/<名字>/model_<步数>.pt
- 曲线数据：每个迭代都写进 runs/<名字>/metrics.csv，用 plot.py 画图

用法：
    uv run python src/rl01/train.py --iterations 5000
    uv run python src/rl01/train.py --iterations 20 --num-envs 128 --name smoke    # 冒烟测试
    uv run python src/rl01/train.py --resume runs/<名字>/model_2500.pt --iterations 5000
"""
import argparse
import csv
import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from rl01.env import N_ACT, N_CRITIC_EXTRA, N_OBS, EnvCfg, Go2Env, STAIR_SCAN, STAIR_STAGES
from rl01.ppo import PPO, ActorCritic
from rl01.symmetry import mirror_action, mirror_obs

RUNS = Path(__file__).resolve().parents[2] / "runs"
FIELDS = ["iteration", "time_s", "fps", "reward_step", "ep_return", "ep_len", "track_lin", "track_ang", "vx", "cmd_vx",
          "falls", "timeouts", "vmax", "surrogate", "value", "kl", "lr", "std", "foot_out_mm", "foot_abs_mm", "swing_z_cm", "sym", "tilt_deg", "rate_xy", "lo_swing_cm", "stage", "stair_success", "stair_done", "front_land", "rear_land"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w-diagonal", type=float, default=0.0)
    ap.add_argument("--stair-front-bonus", type=float, default=0.0)
    ap.add_argument("--stair-rear-bonus", type=float, default=0.0)
    ap.add_argument("--track-sigma", type=float, default=0.25)
    ap.add_argument("--stairs", action="store_true", help="楼梯课程训练")
    ap.add_argument("--log-every", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--iterations", type=int, default=5000, help="总迭代数（'步'）")
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=24, help="每次迭代每个环境走多少步")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--name", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--init-from", default=None, help="只载入这个检查点的网络权重和指令上限，作为新运行的起点（迭代从 0 开始，优化器重置）")
    ap.add_argument("--lr", type=float, default=1e-3, help="初始学习率（自适应 KL 会再调整；微调时用小一点）")
    ap.add_argument("--sym-coef", type=float, default=0.0, help="镜像对称损失的系数，0 关闭")
    ap.add_argument("--w-foot-width", type=float, default=None, help="覆盖足端横向惩罚的权重")
    ap.add_argument("--w-foot-clear", type=float, default=None, help="覆盖摆动脚离地高度惩罚的权重（0 关闭）")
    ap.add_argument("--clear-target", type=float, default=None, help="摆动脚中心的目标高度 m")
    ap.add_argument("--speed-clear", action="store_true", help="启用低速步态奖励：摆动脚目标高度随指令速度变化 + 上界 + 低速时加重横滚/俯仰惩罚")
    ap.add_argument("--w-foot-cap", type=float, default=None)
    ap.add_argument("--cap-margin", type=float, default=None)
    ap.add_argument("--w-orient-low", type=float, default=None, help="低速时额外叠加的 orientation 权重（负数）")
    ap.add_argument("--w-ang-vel-low", type=float, default=None, help="低速时额外叠加的 ang_vel_xy 权重（负数）")
    ap.add_argument("--vx-max", type=float, default=2.0)
    ap.add_argument("--init-vmax", type=float, default=0.5)
    ap.add_argument("--push", type=float, default=0.6, help="随机推力的最大速度冲击 m/s，0 关闭")
    ap.add_argument("--no-noise", action="store_true")
    args = ap.parse_args()

    total = args.iterations
    log_every = args.log_every or max(1, round(total * 0.025))
    save_every = args.save_every or max(1, round(total * 0.10))
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = EnvCfg(w_diagonal=args.w_diagonal, stair_front_bonus=args.stair_front_bonus, stair_rear_bonus=args.stair_rear_bonus, track_sigma=args.track_sigma, stairs=args.stairs, episode_s=45.0 if args.stairs else 20.0, vx_max=args.vx_max, init_vmax=args.init_vmax, push_vel=args.push, noise=not args.no_noise)
    if args.w_foot_width is not None:
        cfg.w_foot_width = args.w_foot_width
    if args.w_foot_clear is not None:
        cfg.w_foot_clear = args.w_foot_clear
    if args.clear_target is not None:
        cfg.foot_clear_target = args.clear_target
    cfg.speed_clear = args.speed_clear
    for a_, f_ in (("w_foot_cap", "w_foot_cap"), ("cap_margin", "cap_margin"), ("w_orient_low", "w_orient_low"), ("w_ang_vel_low", "w_ang_vel_low")):
        if getattr(args, a_) is not None:
            setattr(cfg, f_, getattr(args, a_))
    env = Go2Env(args.num_envs, cfg, seed=args.seed)
    critic_dim = N_OBS + N_CRITIC_EXTRA + (STAIR_SCAN if args.stairs else 0)
    ac = ActorCritic(N_OBS, critic_dim, N_ACT).to(dev)
    ppo = PPO(ac, lr=args.lr, mirror=(mirror_obs, mirror_action) if args.sym_coef > 0 else None, sym_coef=args.sym_coef)

    start = 0
    if args.init_from:
        ck0 = torch.load(args.init_from, map_location=dev)
        state = ck0["model"]
        if args.stairs:
            state = dict(state)
            old = state["critic.0.weight"]
            expanded = ac.state_dict()["critic.0.weight"].clone().zero_()
            expanded[:, :old.shape[1]] = old
            state["critic.0.weight"] = expanded
        ac.load_state_dict(state)
        env.vmax = min(ck0["vmax"], cfg.vx_max)
        print(f"从 {args.init_from} 载入网络权重（微调，迭代从 0 开始）")
    if args.resume:
        ck = torch.load(args.resume, map_location=dev)
        ac.load_state_dict(ck["model"])
        ppo.opt.load_state_dict(ck["optimizer"])
        ppo.lr, env.vmax, start = ck["lr"], ck["vmax"], ck["iteration"]
        if args.stairs:
            env.stage = ck["stair_stage"]
            env.reset_idx(np.arange(env.N))
            env._compute_obs()
        run_dir = Path(args.resume).parent
    else:
        run_dir = RUNS / (args.name or datetime.now().strftime("%Y%m%d_%H%M%S"))
        run_dir.mkdir(parents=True, exist_ok=False)
    config_path = run_dir / "config.json"
    if args.resume:
        old_cfg = json.loads(config_path.read_text())["env"]
        if old_cfg != json.loads(json.dumps(cfg.as_dict())):
            raise ValueError("恢复配置与原运行不同，拒绝覆盖；请使用原参数")
    else:
        config_path.write_text(json.dumps({"args": vars(args), "env": cfg.as_dict()}, indent=2, ensure_ascii=False))
    csv_path = run_dir / "metrics.csv"
    if args.resume and csv_path.exists():
        with csv_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
            w.writerows(r for r in rows if int(r["iteration"]) <= start)
    new_csv = not csv_path.exists()
    csv_file = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
    if new_csv:
        writer.writeheader()

    print(f"运行目录 {run_dir}")
    print(f"设备 {dev}  环境数 {args.num_envs}  每次迭代 {args.steps} 步 = {args.num_envs * args.steps:,} 样本")
    print(f"总迭代 {total}：每 {log_every} 步输出一次（2.5%），每 {save_every} 步保存一次（10%）")

    T, N = args.steps, args.num_envs
    z = lambda *s: torch.zeros(T, N, *s, device=dev)
    st = dict(obs=z(N_OBS), cobs=z(critic_dim), act=z(N_ACT), logp=z(), val=z(), rew=z(), done=z(),
              mu=z(N_ACT), sigma=z(N_ACT))
    recent = deque(maxlen=40)  # 课程判断用：只看最近 40 个迭代（不能累计到升级为止，否则早期的差表现一直拖着平均值）
    pending = []  # 两次输出之间的逐迭代指标，输出时取平均
    t_start = time.time()
    env.pop_stats()

    for it in range(start, total):
        t0 = time.time()
        o, co = env.get_obs()
        for t in range(T):
            ot, cot = torch.as_tensor(o, device=dev), torch.as_tensor(co, device=dev)
            a, logp, val, mu, sig = ac.act(ot, cot)
            rew, done, timeout = env.step(a.cpu().numpy())
            rew_t = torch.as_tensor(rew, dtype=torch.float32, device=dev)
            rew_t += ppo.gamma * val * torch.as_tensor(timeout, dtype=torch.float32, device=dev)  # 超时不是真的失败：用价值做 bootstrap
            st["obs"][t], st["cobs"][t], st["act"][t], st["logp"][t], st["val"][t] = ot, cot, a, logp, val
            st["rew"][t], st["done"][t], st["mu"][t], st["sigma"][t] = rew_t, torch.as_tensor(done, dtype=torch.float32, device=dev), mu, sig
            o, co = env.get_obs()
            if t == 0:
                step_rew_sum = float(rew.mean())
            else:
                step_rew_sum += float(rew.mean())
        with torch.no_grad():
            last_val = ac.value(torch.as_tensor(co, device=dev))
        loss = ppo.update(st, last_val)

        s = env.pop_stats()
        n = max(s["steps"], 1)
        recent.append(s)
        win = dict(ep_len=[l for r in recent for l in r["ep_len"]], rel_err_sum=sum(r["rel_err_sum"] for r in recent),
                   n_mov=sum(r["n_mov"] for r in recent))
        if args.stairs:
            completed = sum(r["stair_current_done"] for r in recent)
            succeeded = sum(r["stair_current_success"] for r in recent)
            if completed >= 128 and succeeded / completed >= 0.70 and env.stage < len(STAIR_STAGES) - 1:
                env.stage += 1
                print(f"楼梯课程升级: {STAIR_STAGES[env.stage]}，上一档 {succeeded}/{completed}", flush=True)
                recent.clear()
        elif len(recent) == recent.maxlen and env.update_curriculum(win):
            recent.clear()  # 升级后重新积累，用新的指令范围重新判断
        row = dict(
            stage=env.stage if args.stairs else -1, stair_success=s["stair_success"], stair_done=s["stair_done"], front_land=s["front_land"], rear_land=s["rear_land"],
            iteration=it + 1, time_s=round(time.time() - t_start, 1), fps=round(T * N / (time.time() - t0)),
            reward_step=step_rew_sum / T, ep_return=float(np.mean(s["ep_return"])) if s["ep_return"] else float("nan"),
            ep_len=float(np.mean(s["ep_len"])) if s["ep_len"] else float("nan"),
            track_lin=s["track_lin"] / n, track_ang=s["track_ang"] / n, vx=s["vx"] / n, cmd_vx=s["cmd_vx"] / n,
            falls=s["falls"], timeouts=s["timeouts"], vmax=env.vmax, surrogate=loss["surrogate"], value=loss["value"],
            kl=loss["kl"], lr=loss["lr"], std=ac.std.mean().item(), foot_out_mm=1000 * s["foot_out_sum"] / n,
            foot_abs_mm=1000 * s["foot_abs_sum"] / n, sym=loss["sym"],
            tilt_deg=float(np.degrees(np.sqrt(s["tilt2"] / n))), rate_xy=float(np.sqrt(s["rate2"] / n)),
            lo_swing_cm=100 * s["lo_swing_z"] / s["lo_swing_n"] if s["lo_swing_n"] else float("nan"),
            swing_z_cm=100 * s["swing_z_sum"] / s["swing_n"] if s["swing_n"] else float("nan"),
        )
        writer.writerow(row)
        csv_file.flush()
        pending.append(row)

        done_it = it + 1
        saved = None
        if done_it % save_every == 0 or done_it == total:
            saved = run_dir / f"model_{done_it}.pt"
            torch.save(dict(model=ac.state_dict(), optimizer=ppo.opt.state_dict(), iteration=done_it, vmax=env.vmax,
                            lr=ppo.lr, env_cfg=cfg.as_dict(), stair_stage=env.stage if args.stairs else None), saved)
        if done_it % log_every == 0 or done_it == total:
            avg = lambda k: float(np.nanmean([r[k] for r in pending])) if not all(np.isnan(r[k]) for r in pending) else float("nan")
            print(f"[{done_it:>6}/{total} {100 * done_it / total:5.1f}%] 用时 {row['time_s'] / 60:6.1f} 分  {avg('fps'):6.0f} 样本/秒\n"
                  f"    回报/步 {avg('reward_step'):+.4f}  回合回报 {avg('ep_return'):+.2f}  存活 {avg('ep_len'):6.0f}/{env.max_steps} 步  "
                  f"摔倒 {sum(r['falls'] for r in pending)}  超时 {sum(r['timeouts'] for r in pending)}\n"
                  f"    线速度跟踪 {avg('track_lin'):.3f}  角速度跟踪 {avg('track_ang'):.3f}  实际 vx {avg('vx'):+.2f} / 指令 vx {avg('cmd_vx'):.2f}  "
                  f"指令上限 {env.vmax:.2f}/{cfg.vx_max:.1f} m/s\n"
                  f"    价值损失 {avg('value'):.4f}  策略损失 {avg('surrogate'):+.4f}  KL {avg('kl'):.4f}  学习率 {row['lr']:.2e}  动作标准差 {row['std']:.3f}\n"
                  f"    足端外偏 {avg('foot_out_mm'):+.1f} mm（绝对值 {avg('foot_abs_mm'):.1f}）  对称损失 {avg('sym'):.4f}  摆动脚离地 {avg('swing_z_cm'):.1f} cm（<0.8 m/s：{avg('lo_swing_cm'):.1f} cm）\n"
                  f"    晃动：倾角 RMS（横滚+俯仰合成）{avg('tilt_deg'):.2f}°  横滚/俯仰角速度 RMS（合成）{avg('rate_xy'):.3f} rad/s"
                  + (f"\n    楼梯 {STAIR_STAGES[env.stage][0]*100:.0f} cm / {STAIR_STAGES[env.stage][1]} 级；本段成功 {sum(r['stair_success'] for r in pending)}/{sum(r['stair_done'] for r in pending)}；落脚前腿 {sum(r['front_land'] for r in pending)} 后腿 {sum(r['rear_land'] for r in pending)}" if args.stairs else "")
                  + (f"\n    已保存 {saved.name}" if saved else ""), flush=True)
            pending = []
    csv_file.close()
    print(f"训练完成，用时 {(time.time() - t_start) / 60:.1f} 分。曲线：uv run python src/rl01/plot.py {run_dir}")


if __name__ == "__main__":
    main()
