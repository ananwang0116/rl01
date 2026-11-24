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
import signal
import shutil
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
FIELDS = ["straight_heading_deg", "cross_vel_mps", "heading_tracking", "reverse_scale", "lateral_scale", "yaw_scale", "reverse_height_cm", "turn_height_cm", "turn_lift_fraction", "turn_drag", "gait_score", "direction_scale", "heading_deg", "stop_speed", "reverse_err", "lateral_err", "yaw_err", "iteration", "time_s", "fps", "reward_step", "ep_return", "ep_len", "track_lin", "track_ang", "vx", "cmd_vx",
          "falls", "timeouts", "vmax", "surrogate", "value", "kl", "lr", "std", "foot_out_mm", "foot_abs_mm", "swing_z_cm", "sym", "tilt_deg", "rate_xy", "lo_swing_cm", "stage", "stair_success", "stair_done", "front_land", "rear_land", "stair_y_cm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--omni', action='store_true', help='全方向+明确停走重启采样')
    ap.add_argument('--heading-obs', action='store_true', help='追加1维航向误差，45->46维')
    ap.add_argument('--relative-tracking', action='store_true')
    ap.add_argument('--heading-bound', type=float, default=0.0)
    ap.add_argument('--maneuver-gait', action='store_true')
    ap.add_argument('--gait-phase-obs', action='store_true')
    ap.add_argument('--gait-frequency', type=float, default=1/0.44)
    ap.add_argument('--gait-frequency-speed', type=float, default=0.0)
    ap.add_argument('--turn-radius-range', type=float, nargs=2, default=[0.8,2.0])
    ap.add_argument('--turn-step-height', type=float, default=0.04)
    ap.add_argument('--turn-focus', action='store_true')
    ap.add_argument('--maneuver-height-target', type=float, default=0.30)
    ap.add_argument('--maneuver-height-floor', type=float, default=0.28)
    ap.add_argument('--w-maneuver-gait', type=float, default=0.0)
    ap.add_argument('--w-maneuver-trajectory', type=float, default=0.0)
    ap.add_argument('--w-maneuver-height', type=float, default=0.0)
    ap.add_argument('--w-maneuver-pose', type=float, default=0.0)
    ap.add_argument('--w-maneuver-slide', type=float, default=0.0)
    ap.add_argument('--init-std', type=float, default=None, help='微调时显式重设探索std')
    ap.add_argument('--reverse-max', type=float, default=1.0)
    ap.add_argument('--vy-range', type=float, default=0.8)
    ap.add_argument('--yaw-range', type=float, default=1.0)
    ap.add_argument('--direction-init', type=float, default=0.2)
    ap.add_argument('--stand-prob', type=float, default=0.1)
    ap.add_argument('--w-heading', type=float, default=0.0)
    ap.add_argument('--w-track-heading', type=float, default=0.0)
    ap.add_argument('--heading-sigma', type=float, default=0.005)
    ap.add_argument('--w-cross-vel', type=float, default=0.0)
    ap.add_argument('--w-stop-vel', type=float, default=0.0)
    ap.add_argument('--w-stop-ang', type=float, default=0.0)
    ap.add_argument('--nthread', type=int, default=8)
    ap.add_argument('--torch-threads', type=int, default=2)
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    ap.add_argument('--max-seconds', type=float, default=3600, help='段落截止时间，到期完整保存')
    ap.add_argument("--w-diagonal", type=float, default=0.0)
    ap.add_argument("--stair-front-bonus", type=float, default=0.0)
    ap.add_argument("--stair-rear-bonus", type=float, default=0.0)
    ap.add_argument("--track-sigma", type=float, default=0.25)
    ap.add_argument("--stairs", action="store_true", help="楼梯课程训练")
    ap.add_argument("--flat-frac", type=float, default=0.0, help="stairs 模式下混入平地环境的比例")
    ap.add_argument("--stair-mix-prev", type=float, default=0.0, help="重置楼梯环境时采样已解锁低档的概率")
    ap.add_argument("--stair-top", type=float, default=1.0, help="顶平台长度 m")
    ap.add_argument("--stair-top-steps", type=int, default=25, help="顶平台达标持续步数 (50Hz)")
    ap.add_argument("--riser-clear-margin", type=float, default=0.0, help="爬梯前方台阶抬脚余量 m")
    ap.add_argument("--obs-taps", type=int, nargs="+", default=[0], help="actor 观测历史帧偏移（0 必须在首位）")
    ap.add_argument("--stair-all-stages", action="store_true", help="楼梯环境同时涵盖 5/10/15 cm × 5/10/15/20 级")
    ap.add_argument("--stair-speed-curriculum", action="store_true", help="楼梯分高度速度课程 (0.25 -> stair_vx_max)")
    ap.add_argument("--stair-speed-ladder", action="store_true", help="平地按 0.3..2.0 阶梯+连续斜坡采样，楼梯上限不超过 stair_vx_max")
    ap.add_argument("--stair-vx-max", type=float, default=1.0, help="爬楼梯指令速度上限 m/s")
    ap.add_argument("--stair-init-vmax", type=float, default=0.5, help="楼梯速度课程起点 m/s")
    ap.add_argument("--stair-lateral-limit", type=float, default=0.42, help="楼梯机身横向偏移 |y| 终止上限 m")
    ap.add_argument("--w-stair-y", type=float, default=0.0, help="楼梯世界 y 偏离平方惩罚权重（负数）")
    ap.add_argument("--w-stair-yaw", type=float, default=0.0, help="楼梯世界航向偏离平方惩罚权重（负数）")
    ap.add_argument("--ang-sigma", type=float, default=0.25, help="track_ang 指数分母")
    ap.add_argument("--cmd-slew-rate", type=float, default=0.0, help="指令 vx 连续斜坡变化速率 m/s^2 (0=阶跃)")
    ap.add_argument("--cmd-resample-s", type=float, default=5.0, help="指令重采样间隔 s")
    ap.add_argument("--friction-range", type=float, nargs=2, default=[0.4, 1.2])
    ap.add_argument("--mass-range", type=float, nargs=2, default=[-1.0, 1.5])
    ap.add_argument("--kp-range", type=float, nargs=2, default=[1.0, 1.0])
    ap.add_argument("--kd-range", type=float, nargs=2, default=[1.0, 1.0])
    ap.add_argument("--damping-range", type=float, nargs=2, default=[1.0, 1.0])
    ap.add_argument("--com-range", type=float, default=0.0)
    ap.add_argument("--push-ang", type=float, default=0.0, help="随机角速度冲击上限 rad/s")
    ap.add_argument("--push-z", type=float, default=0.0, help="随机垂向速度冲击上限 m/s")
    ap.add_argument("--push-interval", type=float, default=5.0, help="平均推力间隔 s")
    ap.add_argument("--w-track-lin", type=float, default=None)
    ap.add_argument("--w-track-ang", type=float, default=None)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--cap-init-std", type=float, default=None, help="--init-from 时把动作标准差上限裁剪到该值")
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
    if args.resume:
        original_args = json.loads((Path(args.resume).parent / 'config.json').read_text())['args']
        for key in ('sym_coef', 'entropy_coef'):
            if key in original_args:
                setattr(args, key, original_args[key])
    if args.omni and args.stairs:
        ap.error('--omni 仅用于平地，不更改楼梯的前进命令分布')
    if not 0 < args.direction_init <= 1 or not 0 <= args.stand_prob < 1:
        ap.error('direction-init 须在 (0,1]，stand-prob 须在 [0,1)')
    torch.set_num_threads(args.torch_threads)
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print(f'收到信号 {signum}，将在本轮更新后保存并退出', flush=True)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    total = args.iterations
    log_every = args.log_every or max(1, round(total * 0.025))
    save_every = args.save_every or max(1, round(total * 0.10))
    torch.manual_seed(args.seed)
    dev = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)

    obs_taps = tuple(args.obs_taps)
    assert obs_taps[0] == 0, "obs_taps 首项必须为 0（当前帧）"
    cfg = EnvCfg(
        omni=args.omni, heading_obs=args.heading_obs, reverse_max=args.reverse_max if args.omni else 0.0,
        relative_tracking=args.relative_tracking, heading_bound=args.heading_bound,
        maneuver_gait=args.maneuver_gait, gait_phase_obs=args.gait_phase_obs,
        gait_frequency=args.gait_frequency, gait_frequency_speed=args.gait_frequency_speed,
        turn_radius_range=tuple(args.turn_radius_range), turn_step_height=args.turn_step_height, turn_focus=args.turn_focus,
        maneuver_height_target=args.maneuver_height_target, maneuver_height_floor=args.maneuver_height_floor,
        w_maneuver_gait=args.w_maneuver_gait, w_maneuver_trajectory=args.w_maneuver_trajectory,
        w_maneuver_height=args.w_maneuver_height, w_maneuver_pose=args.w_maneuver_pose, w_maneuver_slide=args.w_maneuver_slide,
        vy_range=args.vy_range if args.omni else 0.0, yaw_range=args.yaw_range if args.omni else 0.0,
        direction_init=args.direction_init, stand_prob=args.stand_prob,
        w_heading=args.w_heading, w_cross_vel=args.w_cross_vel, w_stop_vel=args.w_stop_vel, w_stop_ang=args.w_stop_ang,
        w_track_heading=args.w_track_heading, heading_sigma=args.heading_sigma,
        w_diagonal=args.w_diagonal, stair_front_bonus=args.stair_front_bonus, stair_rear_bonus=args.stair_rear_bonus,
        track_sigma=args.track_sigma, ang_sigma=args.ang_sigma, stairs=args.stairs, flat_frac=args.flat_frac, stair_mix_prev=args.stair_mix_prev,
        stair_all_stages=args.stair_all_stages, stair_speed_curriculum=args.stair_speed_curriculum,
        stair_speed_ladder=args.stair_speed_ladder, stair_vx_max=args.stair_vx_max, stair_init_vmax=args.stair_init_vmax,
        stair_lateral_limit=args.stair_lateral_limit, w_stair_y=args.w_stair_y, w_stair_yaw=args.w_stair_yaw,
        cmd_slew_rate=args.cmd_slew_rate, cmd_resample_s=args.cmd_resample_s, stair_top=args.stair_top, stair_top_steps=args.stair_top_steps,
        riser_clear_margin=args.riser_clear_margin, obs_taps=obs_taps, episode_s=45.0 if args.stairs else 20.0,
        vx_max=args.vx_max, init_vmax=args.init_vmax, push_vel=args.push, push_ang=args.push_ang, push_z=args.push_z, push_interval_s=args.push_interval, noise=not args.no_noise,
        friction_range=tuple(args.friction_range), mass_range=tuple(args.mass_range),
        kp_range=tuple(args.kp_range), kd_range=tuple(args.kd_range),
        damping_range=tuple(args.damping_range), com_range=args.com_range,
    )
    if args.w_track_lin is not None:
        cfg.w_track_lin = args.w_track_lin
    if args.w_track_ang is not None:
        cfg.w_track_ang = args.w_track_ang
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
    env = Go2Env(args.num_envs, cfg, seed=args.seed, nthread=args.nthread)
    actor_dim = env.n_obs * len(cfg.obs_taps)
    critic_dim = env.n_obs + N_CRITIC_EXTRA + (STAIR_SCAN if args.stairs else 0)
    ac = ActorCritic(actor_dim, critic_dim, N_ACT).to(dev)
    ppo = PPO(ac, lr=args.lr, entropy_coef=args.entropy_coef, mirror=(lambda o: mirror_obs(o, env.n_obs), mirror_action) if args.sym_coef > 0 else None, sym_coef=args.sym_coef)

    start = 0
    if args.init_from:
        ck0 = torch.load(args.init_from, map_location=dev)
        state = dict(ck0["model"])
        for key in ("actor.0.weight", "critic.0.weight"):
            old = state[key]
            cur = ac.state_dict()[key]
            if old.shape[1] != cur.shape[1]:
                expanded = cur.clone().zero_()
                old_frame = N_OBS + int(ck0['env_cfg'].get('heading_obs', False)) + 2*int(ck0['env_cfg'].get('gait_phase_obs', False))
                if key.startswith('critic'):
                    expanded[:, :min(old_frame, env.n_obs)] = old[:, :min(old_frame, env.n_obs)]
                    extra = min(old.shape[1] - old_frame, cur.shape[1] - env.n_obs)
                    expanded[:, env.n_obs:env.n_obs + extra] = old[:, old_frame:old_frame + extra]
                else:
                    frames = min(old.shape[1] // old_frame, len(cfg.obs_taps))
                    for f in range(frames):
                        width = min(old_frame, env.n_obs)
                        expanded[:, f*env.n_obs:f*env.n_obs+width] = old[:, f*old_frame:f*old_frame+width]
                state[key] = expanded
        ac.load_state_dict(state)
        if args.cap_init_std is not None:
            with torch.no_grad():
                ac.std.clamp_(max=args.cap_init_std)
        if args.init_std is not None:
            with torch.no_grad():
                ac.std.fill_(args.init_std)
        env.vmax = max(min(ck0["vmax"], cfg.vx_max), cfg.init_vmax)
        if cfg.omni:
            old_scales = ck0.get('direction_scales',[ck0.get('direction_scale',cfg.direction_init)]*3)
            env.direction_scales = np.maximum(np.minimum(np.array(old_scales,dtype=float),1.0),cfg.direction_init)
            env.direction_scale = float(env.direction_scales.min())
        env._resample_cmd(np.arange(env.N), snap=True)
        env._compute_obs(shift=False)
        print(f"从 {args.init_from} 载入网络权重（微调，迭代从 0 开始）")
    if args.resume:
        ck = torch.load(args.resume, map_location=dev)
        ac.load_state_dict(ck["model"])
        ppo.opt.load_state_dict(ck["optimizer"])
        ppo.lr, env.vmax, start = ck["lr"], ck["vmax"], ck["iteration"]
        for group in ppo.opt.param_groups:
            group['lr'] = ppo.lr
        env.direction_scale = ck.get('direction_scale', cfg.direction_init)
        env.direction_scales = np.array(ck.get('direction_scales',[env.direction_scale]*3),dtype=float)
        env._resample_cmd(np.arange(env.N), snap=True)
        env._compute_obs(shift=False)
        if args.stairs:
            env.stage = ck["stair_stage"]
            if "stair_vmax" in ck and ck["stair_vmax"] is not None:
                env.stair_vmax = np.array(ck["stair_vmax"], dtype=np.float64)
            env.reset_idx(np.arange(env.N))
            env._compute_obs(shift=False)
        run_dir = Path(args.resume).parent
    else:
        run_dir = RUNS / (args.name or datetime.now().strftime("%Y%m%d_%H%M%S"))
        run_dir.mkdir(parents=True, exist_ok=False)
    config_path = run_dir / "config.json"
    if args.resume:
        saved_config = json.loads(config_path.read_text())
        old_cfg = json.loads(json.dumps(EnvCfg(**saved_config['env']).as_dict()))
        if old_cfg != json.loads(json.dumps(cfg.as_dict())):
            raise ValueError("恢复配置与原运行不同，拒绝覆盖；请使用原参数")
        saved_config.setdefault('resume_history', []).append({'checkpoint': args.resume, 'iterations': total})
        saved_config['args']['iterations'] = total
        config_path.write_text(json.dumps(saved_config, indent=2, ensure_ascii=False))
    else:
        config_path.write_text(json.dumps({"args": vars(args), "env": cfg.as_dict()}, indent=2, ensure_ascii=False))
    source_dir = run_dir / ('source' if not args.resume else f'source_resume_{start}')
    source_dir.mkdir(exist_ok=True)
    for filename in ('env.py', 'train.py', 'train_maneuver.py', 'ppo.py', 'symmetry.py'):
        shutil.copy2(Path(__file__).with_name(filename), source_dir / filename)
    csv_path = run_dir / "metrics.csv"
    if args.resume and csv_path.exists():
        with csv_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
            w.writerows({k:r.get(k,'nan') for k in FIELDS} for r in rows if int(r["iteration"]) <= start)
    new_csv = not csv_path.exists()
    csv_file = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
    if new_csv:
        writer.writeheader()

    print(f"运行目录 {run_dir}")
    print(f"设备 {dev}  环境数 {args.num_envs}  每次迭代 {args.steps} 步 = {args.num_envs * args.steps:,} 样本  actor_dim={actor_dim}")
    print(f"总迭代 {total}：每 {log_every} 步输出一次（2.5%），每 {save_every} 步保存一次（10%）")

    T, N = args.steps, args.num_envs
    z = lambda *s: torch.zeros(T, N, *s, device=dev)
    st = dict(obs=z(actor_dim), cobs=z(critic_dim), act=z(N_ACT), logp=z(), val=z(), rew=z(), done=z(),
              mu=z(N_ACT), sigma=z(N_ACT))
    recent = deque(maxlen=40)  # 课程判断用：只看最近 40 个迭代（不能累计到升级为止，否则早期的差表现一直拖着平均值）
    direction_recent = deque(maxlen=80)
    flat_recent = deque(maxlen=40)
    stair_h_recent = [deque(maxlen=40) for _ in range(3)]
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
        direction_recent.append(s)
        if len(direction_recent) == direction_recent.maxlen and env.update_direction_curriculum(direction_recent):
            print(f'全方向课程升级: 后退/侧移/转向 {env.direction_scales.tolist()}，各方向稳定段误差<0.45且存活>85%', flush=True)
            direction_recent.clear()
        flat_recent.append(s)
        for g in range(3):
            stair_h_recent[g].append(s)
        win = dict(ep_len=[l for r in recent for l in r["ep_len"]], rel_err_sum=sum(r["rel_err_sum"] for r in recent),
                   n_mov=sum(r["n_mov"] for r in recent))
        if args.stairs:
            if not cfg.stair_all_stages:
                completed = sum(r["stair_current_done"] for r in recent)
                succeeded = sum(r["stair_current_success"] for r in recent)
                if completed >= 128 and succeeded / completed >= 0.70 and env.stage < len(STAIR_STAGES) - 1:
                    env.stage += 1
                    print(f"楼梯课程升级: {STAIR_STAGES[env.stage]}，上一档 {succeeded}/{completed}", flush=True)
                    recent.clear()
            if cfg.stair_speed_curriculum:
                for g, h_cm in enumerate((5, 10, 15)):
                    if env.stair_vmax[g] >= cfg.stair_vx_max:
                        continue
                    done_g = sum(r["stair_h_done"][g] for r in stair_h_recent[g])
                    succ_g = sum(r["stair_h_success"][g] for r in stair_h_recent[g])
                    nmov_g = sum(r["stair_h_n_mov"][g] for r in stair_h_recent[g])
                    err_g = sum(r["stair_h_err_sum"][g] for r in stair_h_recent[g])
                    if done_g >= 64 and nmov_g >= 4000 and succ_g / done_g >= 0.70 and err_g / nmov_g < 0.30:
                        env.stair_vmax[g] = min(env.stair_vmax[g] + cfg.vmax_step, cfg.stair_vx_max)
                        print(f"楼梯速度课程升级 ({h_cm} cm): vmax={env.stair_vmax[g]:.2f}/{cfg.stair_vx_max:.2f} (成功 {succ_g}/{done_g}, 相对误差 {err_g/nmov_g:.2f})", flush=True)
                        stair_h_recent[g].clear()
            if cfg.flat_frac > 0 and len(flat_recent) == flat_recent.maxlen:
                fwin = dict(flat_ep_len=[l for r in flat_recent for l in r["flat_ep_len"]],
                            flat_rel_err_sum=sum(r["flat_rel_err_sum"] for r in flat_recent),
                            flat_n_mov=sum(r["flat_n_mov"] for r in flat_recent))
                if env.update_curriculum(fwin, flat_only=True):
                    print(f"平地速度课程升级: vmax={env.vmax:.2f}", flush=True)
                    flat_recent.clear()
        elif len(recent) == recent.maxlen and env.update_curriculum(win):
            recent.clear()  # 升级后重新积累，用新的指令范围重新判断
        row = dict(
            straight_heading_deg=float(np.degrees(s['straight_heading_abs']/max(1,s['straight_n']))),
            cross_vel_mps=s['cross_vel_abs']/max(1,s['straight_n']), heading_tracking=s['track_heading']/n,
            reverse_scale=env.direction_scales[0], lateral_scale=env.direction_scales[1], yaw_scale=env.direction_scales[2],
            reverse_height_cm=100*s['reverse_height']/max(1,s['reverse_n']),
            turn_height_cm=100*s['turn_height']/max(1,s['turn_n']),
            turn_lift_fraction=s['turn_lift']/max(1,s['turn_n']),
            turn_drag=s['turn_drag']/max(1,s['turn_n']), gait_score=s['gait_score']/max(1,s['gait_n']),
            direction_scale=env.direction_scale, heading_deg=float(np.degrees(s['heading_abs'] / n)),
            stop_speed=s['stop_speed'] / max(1, s['stop_n']),
            reverse_err=s['direction_err'][0] / max(1, s['direction_n'][0]),
            lateral_err=s['direction_err'][1] / max(1, s['direction_n'][1]),
            yaw_err=s['direction_err'][2] / max(1, s['direction_n'][2]),
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
            stair_y_cm=100 * s["stair_y_abs_sum"] / s["stair_steps"] if s["stair_steps"] else float("nan"),
        )
        writer.writerow(row)
        csv_file.flush()
        pending.append(row)

        done_it = it + 1
        ending = stop_requested or time.time() - t_start >= args.max_seconds or done_it == total
        saved = None
        if done_it % save_every == 0 or ending:
            saved = run_dir / f"model_{done_it}.pt"
            torch.save(dict(model=ac.state_dict(), optimizer=ppo.opt.state_dict(), iteration=done_it, vmax=env.vmax,
                            lr=ppo.lr, env_cfg=cfg.as_dict(), stair_stage=env.stage if args.stairs else None,
                            stair_vmax=env.stair_vmax.tolist() if args.stairs else None,
                            direction_scale=env.direction_scale, direction_scales=env.direction_scales.tolist()), saved)
        if done_it % log_every == 0 or ending:
            avg = lambda k: float(np.nanmean([r[k] for r in pending])) if not all(np.isnan(r[k]) for r in pending) else float("nan")
            stair_label = "全阶段 5/10/15cm" if cfg.stair_all_stages else f"{STAIR_STAGES[env.stage][0]*100:.0f} cm / {STAIR_STAGES[env.stage][1]} 级"
            sv_label = f"；梯速上限 [{env.stair_vmax[0]:.2f}, {env.stair_vmax[1]:.2f}, {env.stair_vmax[2]:.2f}]" if cfg.stair_speed_curriculum else ""
            print(f"[{done_it:>6}/{total} {100 * done_it / total:5.1f}%] 用时 {row['time_s'] / 60:6.1f} 分  {avg('fps'):6.0f} 样本/秒\n"
                  f"    回报/步 {avg('reward_step'):+.4f}  回合回报 {avg('ep_return'):+.2f}  存活 {avg('ep_len'):6.0f}/{env.max_steps} 步  "
                  f"摔倒 {sum(r['falls'] for r in pending)}  超时 {sum(r['timeouts'] for r in pending)}\n"
                  f"    线速度跟踪 {avg('track_lin'):.3f}  角速度跟踪 {avg('track_ang'):.3f}  实际 vx {avg('vx'):+.2f} / 指令 vx {avg('cmd_vx'):.2f}  "
                  f"指令上限 {env.vmax:.2f}/{cfg.vx_max:.1f} m/s\n"
                  f"    价值损失 {avg('value'):.4f}  策略损失 {avg('surrogate'):+.4f}  KL {avg('kl'):.4f}  学习率 {row['lr']:.2e}  动作标准差 {row['std']:.3f}\n"
                  f"    足端外偏 {avg('foot_out_mm'):+.1f} mm（绝对值 {avg('foot_abs_mm'):.1f}）  对称损失 {avg('sym'):.4f}  摆动脚离地 {avg('swing_z_cm'):.1f} cm（<0.8 m/s：{avg('lo_swing_cm'):.1f} cm）\n"
                  f"    晃动：倾角 RMS（横滚+俯仰合成）{avg('tilt_deg'):.2f}°  横滚/俯仰角速度 RMS（合成）{avg('rate_xy'):.3f} rad/s"
                  + (f"\n    后退/侧移/转向幅度 {env.direction_scales.tolist()}  航向误差 {avg('heading_deg'):.2f}°  停止速度 {avg('stop_speed'):.3f} m/s  后退/侧移/转向相对误差 {avg('reverse_err'):.2f}/{avg('lateral_err'):.2f}/{avg('yaw_err'):.2f}" if args.omni else '')
                  + (f"\n    专项姿态 后退高度 {avg('reverse_height_cm'):.1f} cm / 转向高度 {avg('turn_height_cm'):.1f} cm  转向抬脚占比 {avg('turn_lift_fraction'):.2f} / 拖脚指标 {avg('turn_drag'):.2f} / 相位分 {avg('gait_score'):.2f}" if cfg.maneuver_gait else '')
                  + (f"\n    直行航向误差 {avg('straight_heading_deg'):.2f}° / 横向速度误差 {avg('cross_vel_mps'):.3f} m/s / 航向跟踪分 {avg('heading_tracking'):.3f}" if cfg.w_track_heading else '')
                  + (f"\n    楼梯 {stair_label}{sv_label}；本段成功 {sum(r['stair_success'] for r in pending)}/{sum(r['stair_done'] for r in pending)}；横向 |y| 均值 {avg('stair_y_cm'):.1f} cm；落脚前腿 {sum(r['front_land'] for r in pending)} 后腿 {sum(r['rear_land'] for r in pending)}" if args.stairs else "")
                  + (f"\n    已保存 {saved.name}" if saved else ""), flush=True)
            pending = []
        if ending:
            break
    csv_file.close()
    env.roller.close()
    status = '本段保存退出' if done_it < total else '训练完成'
    print(f"{status}，迭代 {done_it}/{total}，用时 {(time.time() - t_start) / 60:.1f} 分。曲线：uv run python src/rl01/plot.py {run_dir}", flush=True)


if __name__ == "__main__":
    main()
