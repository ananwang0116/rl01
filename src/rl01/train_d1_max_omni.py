"""D1 Max 全向 PPO：每一段保存可恢复检查点，由独立评估决定是否提高扰动。"""
import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch

from rl01.d1_max_env import JOINT_NAMES, N_ACT, mirror_action
from rl01.d1_max_omni import N_OBS, N_CRITIC_OBS, OmniConfig, OmniEnv, mirror_obs
from rl01.ppo import ActorCritic, PPO

TASK = "d1-max-omni-v2"
CHAMP_TASK = "d1-max-omni-champ-v1"
HIDDEN = (192, 128, 64)
SPEED_LADDER = (.65, 1., 1.5, 2., 2.6, 3., 3.5)


class MirroredActor(torch.nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, obs):
        return .5 * (self.network(obs) + mirror_action(self.network(mirror_obs(obs))))


def make_policy():
    actor_critic = ActorCritic(N_OBS, N_CRITIC_OBS, N_ACT, hidden=HIDDEN, init_std=.23)
    actor_critic.actor = MirroredActor(actor_critic.actor)
    return actor_critic


def read_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    expected = CHAMP_TASK if ckpt["env"].get("champ_gait", False) else TASK
    if ckpt["task"] != expected or ckpt["joint_names"] != JOINT_NAMES:
        raise ValueError(f"checkpoint task or joint mapping mismatch: {path}")
    return ckpt


def load_policy(path):
    ckpt = read_checkpoint(path)
    model = make_policy()
    model.load_state_dict(ckpt["model"])
    return model.actor.eval(), OmniConfig(**ckpt["env"])


def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory
        free, _ = torch.cuda.mem_get_info(0)
        if args.gpu_memory_gib is not None:
            allowed = int(args.gpu_memory_gib * 1024**3)
            if free < allowed:
                raise RuntimeError(f"CUDA free memory {free / 1024**3:.2f} GiB is below the requested cap")
            torch.cuda.set_per_process_memory_fraction(min(1., allowed / total), device=0)
        print(f"cuda_device={torch.cuda.get_device_name(0)} free_gib={free / 1024**3:.2f} "
              f"allocator_limit_gib={args.gpu_memory_gib}", flush=True)
    ckpt = read_checkpoint(args.resume, device) if args.resume else None
    cfg = OmniConfig(**ckpt["env"]) if ckpt else OmniConfig()
    cfg = replace(cfg, robustness=args.robustness)
    if args.corrective_gait:
        cfg = replace(cfg, corrective_gait=True, lateral_lift_m=.062, lateral_gait_hz=1.35)
    if args.champ_gait:
        if ckpt and not cfg.champ_gait:
            raise ValueError("CHAMP changes the gait observations; train from scratch or resume a CHAMP checkpoint")
        cfg = replace(cfg, champ_gait=True, corrective_gait=True)
    if args.champ_ik:
        if not cfg.champ_gait:
            raise ValueError("--champ-ik requires the CHAMP gait")
        cfg = replace(cfg, champ_ik=True)
    if args.champ_refine:
        if not cfg.champ_gait:
            raise ValueError("--champ-refine requires the CHAMP gait")
        cfg = replace(cfg, champ_refine=True)
    if args.vy_turn_only:
        if not cfg.champ_gait:
            raise ValueError("--vy-turn-only requires the CHAMP gait")
        cfg = replace(cfg, vy_turn_only=True)
    if args.speed_stage is not None:
        if not cfg.champ_gait:
            raise ValueError("--speed-stage requires the CHAMP gait")
        speed = SPEED_LADDER[args.speed_stage]
        # Keep wheel-speed headroom for the outer wheel during a moving turn.
        cfg = replace(cfg, max_speed=speed, max_reverse=.65,
                      wheel_speed_limit=max(15., (speed + 1.05) / .096))
    env = OmniEnv(args.num_envs, cfg,
                 seed=args.seed, nthread=args.nthread)
    ac = make_policy().to(device)
    ppo = PPO(ac, lr=args.lr, epochs=4, desired_kl=0, entropy_coef=.002)
    start = 0
    if ckpt:
        ac.load_state_dict(ckpt["model"])
        ppo.opt.load_state_dict(ckpt["optimizer"])
        start = ckpt["iteration"]
        for group in ppo.opt.param_groups:
            group["lr"] = args.lr
    else:
        torch.nn.init.zeros_(ac.actor.network[-1].weight)
        torch.nn.init.zeros_(ac.actor.network[-1].bias)

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = vars(args) | {"output": str(out), "resume": str(args.resume) if args.resume else None,
                            "env": asdict(env.cfg), "start_iteration": start, "joint_names": JOINT_NAMES}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    stopped = False

    def request_stop(_sig, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    begin = time.monotonic()
    shape = (args.steps, args.num_envs)
    widths = {"obs": N_OBS, "cobs": N_CRITIC_OBS, "act": N_ACT, "mu": N_ACT, "sigma": N_ACT}
    storage = {key: torch.empty((*shape, width), device=device) for key, width in widths.items()}
    storage.update({key: torch.empty(shape, device=device) for key in ("logp", "val", "rew", "done")})
    fields = ("iteration", "elapsed_s", "reward", "vx_mae", "vy_mae", "yaw_mae", "tilt_deg",
              "lift_mm", "falls", "timeouts", "fps", "kl", "reward_vx", "reward_vy", "reward_yaw",
              "reward_lift", "reward_posture", "reward_smooth")
    display_tick = check_tick = save_tick = 0

    def tensor(values):
        return torch.as_tensor(values, device=device, dtype=torch.float32)

    print(f"stage_started run={out.name} start={start} scale={env.cfg.robustness:g} "
          f"wall_budget_s={args.max_seconds:g} nenv={args.num_envs} device={args.device}", flush=True)
    try:
        obs, critic = map(tensor, env.observations())
        with (out / "metrics.csv").open("w", newline="") as log:
            writer = csv.DictWriter(log, fieldnames=fields)
            writer.writeheader()
            for it in range(start + 1, start + args.iterations + 1):
                t = time.monotonic()
                accum = dict.fromkeys(fields, 0.0)
                for k in range(args.steps):
                    action, logp, value, mean, std = ac.act(obs, critic)
                    reward, done, info = env.step(action.cpu().numpy())
                    rew = tensor(reward).clone()
                    if info["timeout"].any():
                        ids = torch.as_tensor(np.flatnonzero(info["timeout"]), device=device)
                        with torch.no_grad():
                            terminal = ac.value(tensor(info["terminal_critic"]))
                        rew[ids] += ppo.gamma * terminal
                    samples = dict(obs=obs, cobs=critic, act=action, logp=logp, val=value,
                                   mu=mean, sigma=std, rew=rew, done=tensor(done))
                    for key, sample in samples.items():
                        storage[key][k] = sample
                    obs, critic = map(tensor, env.observations())
                    accum["reward"] += float(reward.mean())
                    for axis in ("vx", "vy", "yaw"):
                        accum[f"{axis}_mae"] += float(np.abs(info[f"{axis}_error"]).mean())
                    accum["tilt_deg"] += float(np.degrees(info["tilt"]).mean())
                    accum["lift_mm"] += float(info["clearance"].max(axis=1).mean() * 1000)
                    accum["falls"] += int(info["fall"].sum())
                    accum["timeouts"] += int(info["timeout"].sum())
                    for key, values in info["terms"].items():
                        accum[f"reward_{key}"] += float(values.mean())

                with torch.no_grad():
                    last = ac.value(critic)
                loss = ppo.update(storage, last)
                if not np.isfinite([accum["reward"], loss["value"], loss["kl"]]).all():
                    raise RuntimeError(f"non-finite optimization result at {it}")
                for key in fields[2:8] + fields[12:]:
                    accum[key] /= args.steps
                elapsed = time.monotonic() - begin
                accum.update(iteration=it, elapsed_s=round(elapsed, 2), kl=loss["kl"],
                             fps=round(args.steps * args.num_envs / (time.monotonic() - t)))
                writer.writerow(accum)
                log.flush()
                progress = min(1., max((it - start) / args.iterations, elapsed / args.max_seconds))
                next_display = int(40 * progress + 1e-9)
                next_check = int(20 * progress + 1e-9)
                next_save = int(10 * progress + 1e-9)
                reason = ("signal" if stopped else "wall-timer" if elapsed >= args.max_seconds else
                          "iterations" if it == start + args.iterations else None)
                if next_check > check_tick:
                    check_tick = next_check
                    fall_rate = accum["falls"] / (args.steps * args.num_envs)
                    healthy = fall_rate <= .02 and accum["tilt_deg"] <= 15
                    print(f"checkpoint_health progress={5 * check_tick}% iteration={it} "
                          f"fall_rate={fall_rate:.4f} tilt_deg={accum['tilt_deg']:.2f} "
                          f"vy_mae={accum['vy_mae']:.3f} yaw_mae={accum['yaw_mae']:.3f} healthy={healthy}",
                          flush=True)
                    if not healthy:
                        reason = "health-check"
                if it == start + 1 or next_display > display_tick or reason:
                    display_tick = next_display
                    print(f"iteration={it} progress={100 * progress:.1f}% wall_s={elapsed:.1f} reward={accum['reward']:.3f} "
                          f"vx={accum['vx_mae']:.3f} vy={accum['vy_mae']:.3f} yaw={accum['yaw_mae']:.3f} "
                          f"lift_mm={accum['lift_mm']:.1f} falls={int(accum['falls'])} "
                          f"timeouts={int(accum['timeouts'])} fps={accum['fps']}", flush=True)
                extra_save = args.save_every and (it - start) % args.save_every == 0
                if reason or next_save > save_tick or extra_save:
                    save_tick = next_save
                    path = out / f"model_{it}.pt"
                    tmp = path.with_suffix(".tmp")
                    torch.save({"task": CHAMP_TASK if env.cfg.champ_gait else TASK,
                                "iteration": it, "joint_names": JOINT_NAMES,
                                "env": asdict(env.cfg), "model": ac.state_dict(),
                                "optimizer": ppo.opt.state_dict(), "lr": ppo.lr, "stop_reason": reason}, tmp)
                    tmp.replace(path)
                    print(f"checkpoint={path}", flush=True)
                if reason:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                        print(f"cuda_peak_allocated_mib={torch.cuda.max_memory_allocated() / 1024**2:.1f} "
                              f"cuda_peak_reserved_mib={torch.cuda.max_memory_reserved() / 1024**2:.1f}", flush=True)
                    print(f"stage_complete iteration={it} wall_s={elapsed:.1f} reason={reason}", flush=True)
                    break
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--num-envs", type=int, default=96)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--nthread", type=int, default=16)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--gpu-memory-gib", type=float, help="CUDA caching allocator cap in GiB")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--robustness", type=float, default=.25)
    p.add_argument("--corrective-gait", action="store_true", help="Use the lateral-focused corrective gait")
    p.add_argument("--champ-gait", action="store_true", help="Train body-triggered alternating diagonal placement")
    p.add_argument("--champ-ik", action="store_true", help="Use model-based wheel-center inverse kinematics")
    p.add_argument("--champ-refine", action="store_true", help="Keep short yaw taps low and smooth the wheel command")
    p.add_argument("--vy-turn-only", action="store_true", help="Train only lateral (vy) and in-place turn (wz) behaviors")
    p.add_argument("--speed-stage", type=int, choices=range(len(SPEED_LADDER)),
                   help="Index into the 0.65 to 3.5 m/s forward-speed curriculum")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--save-every", type=int, help="Optional additional checkpoint interval in PPO iterations")
    p.add_argument("--max-seconds", type=float, required=True)
    a = p.parse_args()
    if (min(a.iterations, a.num_envs, a.steps) <= 0 or not 0 <= a.robustness <= 1 or a.nthread < 0
            or (a.save_every is not None and a.save_every <= 0)):
        p.error("invalid training, thread or robustness range")
    if not np.isfinite(a.max_seconds) or a.max_seconds <= 0 or a.num_envs * a.steps % 4:
        p.error("need a positive finite wall budget and minibatch-divisible rollout")
    if a.gpu_memory_gib is not None and (
        a.device != "cuda" or not np.isfinite(a.gpu_memory_gib) or a.gpu_memory_gib <= 0
    ):
        p.error("gpu-memory-gib is a positive CUDA-only limit")
    run(a)


if __name__ == "__main__":
    main()
