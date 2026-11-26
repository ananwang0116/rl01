"""Bounded first-episode stair entry/exit and stop/restart checks (MuJoCo only).

Example (run sequentially, with no CPU-heavy training on this machine)::
    MUJOCO_GL=egl uv run python tools/eval/stair_transition.py \
        runs/go2_stairs_v9/model_800.pt --stage 11 --vx .5 \
        --case continuous stop_restart --seconds 35 --out runs/stair_transition

Checkpoint observation/history/optional heading and phase contracts are preserved.
The stair direction is world +x. No desired heading is added to legacy policies.
Site touch measures contact, but touch plus tread position does not identify the
specific collision surface. Geometric crossings and qualified touch events are
reported separately. Terminal physics samples are included before auto-reset;
subsequent episodes are excluded. --seconds caps each case; --wall-seconds caps
the whole script. Partial traces/results are saved if the wall budget expires.
"""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from PIL import Image, ImageDraw
import torch

from rl01.env import (EnvCfg, Go2Env, STAIR_STAGES, STAIR_X0, STAIR_TREAD,
                      STAIR_HALF_WIDTH, S_QUAT, S_LIN, S_ANG, quat_to_R,
                      build_train_model)
from rl01.play import load_policy


def distribution(values):
    x = np.asarray(values, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if not x.size:
        return {"n": 0, "mean": None, "rms": None, "p95_abs": None, "max_abs": None}
    return {"n": int(x.size), "mean": float(x.mean()), "rms": float(np.sqrt(np.mean(x*x))),
            "p95_abs": float(np.percentile(np.abs(x), 95)), "max_abs": float(np.abs(x).max())}


def make_cfg(checkpoint, args):
    values = dict(checkpoint["env_cfg"])
    for key in ("friction_range", "mass_range", "kp_range", "kd_range", "damping_range", "obs_taps"):
        if key in values:
            values[key] = tuple(values[key])
    cfg = EnvCfg(**values)
    cfg.stairs, cfg.flat_frac = True, 0.
    cfg.stair_all_stages, cfg.stair_mix_prev = False, 0.
    cfg.episode_s = args.seconds + 2.
    cfg.omni = False  # Fixed evaluation commands replace the reset sampler.
    cfg.flat_cmd_step_prob = 0.0
    cfg.cmd_slew_rate = 0.  # Actual command steps are identical for old/new policies.
    cfg.stair_stop_prob = 0.  # No unused random stop draws during reset sampling.
    cfg.cmd_resample_s = 4.  # Identical unused reset-command lifetime sampling.
    cfg.push_vel, cfg.push_ang, cfg.push_z = args.push, args.push_ang, args.push_z
    cfg.push_interval_s = args.push_interval
    # Fixed clean/stress conditions must not be silently attenuated by a new
    # checkpoint's training robustness curriculum. Training profiles reproduce
    # the saved unlocked strength rather than its original initial value.
    if args.profile in ("clean", "stress"):
        cfg.robustness_curriculum = False
    elif cfg.robustness_curriculum:
        cfg.robustness_init = float(checkpoint.get("robustness_scale", cfg.robustness_init))
    if args.profile == "clean":
        cfg.noise = False
        cfg.friction_range = (.8, .8)
        cfg.mass_range = (0., 0.)
        cfg.kp_range = cfg.kd_range = cfg.damping_range = (1., 1.)
        cfg.com_range = 0.
    elif args.profile == "stress":
        cfg.noise = True
        cfg.friction_range = (.35, 1.25)
        cfg.mass_range = (-1.2, 2.)
        cfg.kp_range = cfg.kd_range = (.82, 1.18)
        cfg.damping_range = (.45, 1.25)
        cfg.com_range = .035
    # Give both stop bouts time to finish on a short staircase; do not shorten
    # the checkpoint's top-platform hold. Success remains the env's touch test.
    cfg.stair_top_steps = max(cfg.stair_top_steps, round((args.stop_seconds + 1.) / .02))
    return cfg


def filmstrip(frames, output):
    if not frames:
        return None
    w, h = frames[0][1].shape[1], frames[0][1].shape[0]
    canvas = Image.new("RGB", (4*w, math.ceil(len(frames)/4)*(h+24)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, pixels) in enumerate(frames):
        x, y = index % 4*w, index // 4*(h+24)
        canvas.paste(Image.fromarray(pixels), (x, y+24))
        draw.text((x+4, y+5), label, fill="black")
    canvas.save(output)
    return str(output)


def finite_or_none(values):
    return [float(x) if np.isfinite(x) else None for x in values]


def evaluate(policy, checkpoint, args, case, folder, deadline):
    cfg = make_cfg(checkpoint, args)
    env = Go2Env(args.num_envs, cfg, seed=args.seed, nthread=min(4, args.num_envs))
    renderer = None
    started = time.monotonic()
    height, count = STAIR_STAGES[args.stage]
    last_edge = STAIR_X0 + (count-1)*STAIR_TREAD
    active = np.ones(env.N, bool)
    outcomes = {key: np.zeros(env.N, bool) for key in ("success", "fall", "timeout")}
    outcome_time = np.full(env.N, np.nan)
    stop_phase = np.zeros(env.N, int)
    stop_until = np.full(env.N, -np.inf)
    stop_times = np.full((env.N, 2), np.nan)
    restart_times = np.full((env.N, 2), np.nan)
    # Per-leg first qualified touch at first/last tread; not an all-feet reach test.
    first_touch = np.full((env.N, 4), np.nan)
    last_touch = np.full((env.N, 4), np.nan)
    geometric_times = np.full((env.N, 2), np.nan)
    frames = {key: [] for key in ("entry", "exit", "stop1", "restart1", "stop2", "restart2")}
    windows = {}
    trace = {key: [] for key in ("time_s", "state", "feet", "touch", "valid", "command", "velocity", "yaw", "roll", "pitch", "knee_low", "push_xy")}
    status = "completed"
    try:
        env.stage = args.stage  # Must precede reset to select the requested model.
        env.reset_idx(np.arange(env.N))
        if args.aligned_start:
            # Isolate uncommanded transition twisting from deliberate recovery
            # of the reset sampler's initial yaw/lateral error. Other reset
            # randomization (joint pose, linear velocity, dynamics) is retained.
            env.state[:, S_QUAT] = [1., 0., 0., 0.]
            env.state[:, 2] = 0.
            env.state[:, S_ANG] = 0.
            env.heading_target[:] = 0.
            env._just_reset[:] = True  # Refill every history tap from aligned obs.
        env._compute_obs(shift=False)
        if not args.no_frames:
            model = build_train_model(visual=True, stair_stage=args.stage, stair_top=cfg.stair_top)
            model.light_castshadow[:] = False
            data = mujoco.MjData(model)
            renderer = mujoco.Renderer(model, 240, 400)
        frame_stride = max(1, round(args.frame_interval/env.dt))
        last_progress = time.monotonic()
        for step in range(round(args.seconds/env.dt)):
            if time.monotonic() >= deadline:
                status = "wall_deadline"
                break
            t = step*env.dt
            feet_before = env.sens_buf[:, -1, :12].reshape(env.N, 4, 3)
            if case == "stop_restart":
                # Spatial triggers guarantee stopping on stairs/platform rather
                # than at an arbitrary time while a slow policy is still below.
                for stop_index, edge in enumerate((STAIR_X0+.05, last_edge+.05)):
                    ready = active & (stop_phase == stop_index*2) & (feet_before[:, :, 0] > edge).all(axis=1)
                    stop_times[ready, stop_index] = t
                    stop_until[ready] = t+args.stop_seconds
                    stop_phase[ready] += 1
                    if ready[0]:
                        windows[f"stop{stop_index+1}"] = step
                    restarting = active & (stop_phase == stop_index*2+1) & (t >= stop_until)
                    restart_times[restarting, stop_index] = t
                    stop_phase[restarting] += 1
                    if restarting[0]:
                        windows[f"restart{stop_index+1}"] = step
            cmd = np.zeros((env.N, 3))
            cmd[:, 0] = np.where(t < stop_until, 0., args.vx)
            env.cmd[:] = env.cmd_target[:] = cmd
            env.cmd_left[:] = 10**9
            env._compute_obs(shift=False)
            obs, _ = env.get_obs()
            if obs.shape[1] != policy.actor[0].in_features:
                raise ValueError(f"observation mismatch: env {obs.shape[1]} vs actor {policy.actor[0].in_features}")
            with torch.inference_mode():
                o = torch.as_tensor(obs)
                action = (policy.dist(o).sample() if args.sampled else policy.actor(o)).numpy()
            _, done, _ = env.step(action)
            # env.state is already reset on done; these buffers retain the actual
            # terminal step and remain valid until the next rollout.
            raw = env.state_buf[:, -1].copy()
            sensor = env.sens_buf[:, -1].copy()
            feet = sensor[:, :12].reshape(env.N, 4, 3)
            touch = sensor[:, 24:28] > .1
            rotation = quat_to_R(raw[:, S_QUAT])
            velocity = np.c_[np.einsum("nji,nj->ni", rotation, raw[:, S_LIN])[:, :2], raw[:, 25]]
            yaw = np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0])
            roll = np.arctan2(rotation[:, 2, 1], rotation[:, 2, 2])
            pitch = np.arcsin(np.clip(-rotation[:, 2, 0], -1., 1.))
            valid = active & np.isfinite(raw).all(axis=1) & np.isfinite(sensor).all(axis=1)
            for level, saved in ((1, first_touch), (count, last_touch)):
                edge = STAIR_X0+(level-1)*STAIR_TREAD
                # Top tread extends to end of configured platform.
                tread_end = edge+ (cfg.stair_top if level == count else STAIR_TREAD)
                qualified = touch & (feet[:, :, 0] >= edge+.01) & (feet[:, :, 0] <= tread_end-.01)
                qualified &= np.abs(feet[:, :, 1]) < STAIR_HALF_WIDTH
                qualified &= np.abs(feet[:, :, 2] - (level*height+.022)) < .035
                fresh = valid[:, None] & qualified & ~np.isfinite(saved)
                saved[fresh] = (step+1)*env.dt
            for index, edge in enumerate((STAIR_X0, last_edge)):
                crossed = valid & (feet[:, :, 0] >= edge).all(axis=1) & ~np.isfinite(geometric_times[:, index])
                geometric_times[crossed, index] = (step+1)*env.dt
            for index, key in enumerate(("success", "fall", "timeout")):
                hit = active & env.last_outcome[index]
                outcomes[key] |= hit
                outcome_time[hit] = (step+1)*env.dt
            knee = sensor[:, 12:24].reshape(env.N, 4, 3)
            ground = np.minimum(env.terrain_height(knee[:, :, 0], knee[:, :, 1]),
                                np.maximum(0, np.floor(knee[:, :, 2]/height)*height))
            values = dict(time_s=(step+1)*env.dt, state=raw, feet=feet, touch=touch,
                          valid=valid, command=cmd, velocity=velocity, yaw=yaw, roll=roll,
                          pitch=pitch, knee_low=(knee[:, :, 2]-ground < .035), push_xy=env.push_delta.copy())
            for key, value in values.items():
                trace[key].append(np.array(value, copy=True))
            # Env 0 only: consecutive 0.08 s images starting just before each
            # riser transition. Absence of exit images means it was not reached.
            if valid[0]:
                if "entry" not in windows and feet[0, :2, 0].max() >= STAIR_X0-.08:
                    windows["entry"] = step
                if "exit" not in windows and feet[0, 2:, 0].max() >= last_edge-.08:
                    windows["exit"] = step
                due = [key for key, start in windows.items() if step >= start and
                       (step-start) % frame_stride == 0 and (step-start)//frame_stride < args.frame_count]
                if renderer is not None and due:
                    mujoco.mj_setState(model, data, raw[0], mujoco.mjtState.mjSTATE_FULLPHYSICS)
                    mujoco.mj_forward(model, data)
                    camera = mujoco.MjvCamera()
                    camera.lookat[:] = [raw[0, 1], raw[0, 2], max(.3, raw[0, 3]-.05)]
                    camera.distance, camera.elevation, camera.azimuth = 1.45, -10, 90
                    renderer.update_scene(data, camera=camera)
                    pixels = renderer.render().copy()
                    for key in due:
                        label = f"{key} {(step+1)*env.dt:.2f}s vx={cmd[0,0]:.1f}"
                        frames[key].append((label, pixels))
            active &= ~done
            if time.monotonic()-last_progress >= 30:
                print(f"progress {case} simulated={(step+1)*env.dt:.1f}s active={active.sum()}/{env.N}", flush=True)
                last_progress = time.monotonic()
            if not active.any():
                break
    finally:
        if renderer is not None:
            renderer.close()
        env.roller.close()
    trace = {key: np.asarray(value) for key, value in trace.items()}
    result = {"checkpoint": args.current_checkpoint, "iteration": checkpoint.get("iteration"),
              "case": case, "stage": args.stage, "height_m": height, "steps": count,
              "num_envs": env.N, "seed": args.seed, "vx": args.vx, "sampled": args.sampled,
              "aligned_start": args.aligned_start,
              "aligned_start_description": "yaw=0, world y=0, angular velocity=0; retain other reset randomness" if args.aligned_start else "checkpoint reset sampler",
              "profile": args.profile, "status": status, "wall_seconds": time.monotonic()-started,
              "effective_robustness_scale": float(env.robustness_scale),
              "top_hold_evaluation": {"checkpoint_steps": checkpoint["env_cfg"].get("stair_top_steps", 25),
                                      "evaluation_steps": cfg.stair_top_steps,
                                      "rule": "max(checkpoint hold, (stop_seconds+1)/dt); identical rule for continuous/stop_restart and all checkpoints"},
              "simulated_seconds": float(trace["time_s"][-1]) if len(trace["time_s"]) else 0.,
              "outcomes": {key: int(value.sum()) for key, value in outcomes.items()},
              "active_at_end": int(active.sum()), "outcome_time_s": finite_or_none(outcome_time),
              "stair_heading_reference": "world +x; yaw=0, not per-episode initial yaw",
              "terminal_samples": "state_buf/sens_buf before reset; only first episodes",
              "randomness_comparison": "fixed reset command sampler; training/stress match distributions, not exact noise/push sequences when observation dimensions differ",
              "qualified_touch_measurement": "measured foot-site touch >0.1 plus foot centre within tread bounds/height; not collision-pair surface identification",
              "geometric_measurement": "all four foot centres cross x edge; not contact or success",
              "knee_low_measurement": "analytic terrain-relative knee height <.035m; not measured collision",
              "per_env": [{"env": i, "first_tread_qualified_touch_s": finite_or_none(first_touch[i]),
                           "last_tread_qualified_touch_s": finite_or_none(last_touch[i]),
                           "all_feet_cross_first_last_edge_s": finite_or_none(geometric_times[i]),
                           "stop_times_s": finite_or_none(stop_times[i]), "restart_times_s": finite_or_none(restart_times[i])}
                          for i in range(env.N)], "regions": {}, "stop_restart": [], "env_cfg": asdict(cfg)}
    if len(trace["time_s"]):
        valid, state, feet = trace["valid"], trace["state"], trace["feet"]
        max_abs_y = [float(np.abs(state[valid[:, i], i, 2]).max()) if valid[:, i].any() else None
                     for i in range(env.N)]
        result["max_abs_y_m"] = distribution([x for x in max_abs_y if x is not None])
        for i, value in enumerate(max_abs_y):
            result["per_env"][i]["max_abs_y_m"] = value
        front = feet[:, :, :2, 0].max(axis=2)
        rear = feet[:, :, 2:, 0].min(axis=2)
        regions = {"approach": front < STAIR_X0-.25,
                   "entry": (front >= STAIR_X0-.25) & (rear < STAIR_X0+.25),
                   "middle": (rear >= STAIR_X0+.25) & (front < last_edge-.25),
                   "exit": (front >= last_edge-.25) & (rear < last_edge+.3),
                   "top": rear >= last_edge+.3}
        for name, region in regions.items():
            mask = valid & region
            result["regions"][name] = {"env_samples": int(mask.sum()),
                "heading_error_deg": distribution(np.degrees(trace["yaw"][mask])),
                "yaw_rate_radps": distribution(trace["velocity"][:, :, 2][mask]),
                "roll_deg": distribution(np.degrees(trace["roll"][mask])),
                "pitch_deg": distribution(np.degrees(trace["pitch"][mask])),
                "lateral_world_m": distribution(state[:, :, 2][mask]),
                "xy_velocity_error_mps": distribution(np.linalg.norm(trace["velocity"][:, :, :2]-trace["command"][:, :, :2], axis=2)[mask]),
                "knee_low_fraction": float(trace["knee_low"][mask].mean()) if mask.any() else None,
                "measured_touch_fraction_per_leg": trace["touch"][mask].mean(axis=0).tolist() if mask.any() else None}
        for index in range(2):
            stop_mask = valid & (trace["time_s"][:, None] >= stop_times[None, :, index]+.5) & (trace["time_s"][:, None] <= restart_times[None, :, index])
            restart_mask = valid & (trace["time_s"][:, None] >= restart_times[None, :, index]) & (trace["time_s"][:, None] <= restart_times[None, :, index]+1.)
            result["stop_restart"].append({"bout": index+1, "stopped_envs": int(np.isfinite(stop_times[:, index]).sum()),
                "restarted_envs": int(np.isfinite(restart_times[:, index]).sum()),
                "stop_speed_after_0_5s_mps": distribution(np.linalg.norm(trace["velocity"][:, :, :2], axis=2)[stop_mask]),
                "restart_first_1s_vx_error_mps": distribution((trace["velocity"][:, :, 0]-args.vx)[restart_mask]),
                "restart_heading_error_deg": distribution(np.degrees(trace["yaw"][restart_mask]))})
        pushed = valid & (np.linalg.norm(trace["push_xy"], axis=2)>0)
        angles = np.arctan2(trace["push_xy"][:, :, 1], trace["push_xy"][:, :, 0])[pushed]
        result["push_events_xy"] = int(pushed.sum())
        result["push_bearing_octants"] = np.histogram(angles, np.linspace(-np.pi, np.pi, 9))[0].tolist()
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/"trace.npz", **trace)
    result["filmstrips"] = {key: filmstrip(value, folder/f"{key}.png") for key, value in frames.items()}
    for key, value in frames.items():
        if value:
            frame_folder = folder/key
            frame_folder.mkdir(exist_ok=True)
            for index, (_, pixels) in enumerate(value):
                Image.fromarray(pixels).save(frame_folder/f"{index:02d}.png")
    (folder/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: result[key] for key in ("case", "status", "outcomes", "active_at_end", "simulated_seconds")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="+")
    parser.add_argument("--stage", type=int, default=2, choices=range(len(STAIR_STAGES)))
    parser.add_argument("--vx", type=float, default=.5)
    parser.add_argument("--seed", type=int, default=719)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=35.)
    parser.add_argument("--wall-seconds", type=float, default=120.)
    parser.add_argument("--case", nargs="+", default=["continuous", "stop_restart"], choices=("continuous", "stop_restart"))
    parser.add_argument("--stop-seconds", type=float, default=1.5)
    parser.add_argument("--profile", choices=("clean", "training", "stress"), default="clean")
    parser.add_argument("--push", type=float, default=0., help="360-degree XY velocity impulse max m/s")
    parser.add_argument("--push-ang", type=float, default=0., help="all-axis angular impulse max rad/s")
    parser.add_argument("--push-z", type=float, default=0., help="vertical velocity impulse max m/s")
    parser.add_argument("--push-interval", type=float, default=4.5)
    parser.add_argument("--sampled", action="store_true")
    parser.add_argument("--aligned-start", action="store_true", help="start exactly along stairs: yaw=0, y=0, angular velocity=0; retain other reset randomness")
    parser.add_argument("--no-frames", action="store_true")
    parser.add_argument("--frame-interval", type=float, default=.08)
    parser.add_argument("--frame-count", type=int, default=12)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not (1 <= args.num_envs <= 64 and 0 < args.seconds <= 120 and args.wall_seconds > 0 and
            0 < args.vx <= 1. and 0 < args.stop_seconds <= 10 and args.push_interval > 0 and
            0 <= min(args.push, args.push_ang, args.push_z) and args.frame_interval > 0 and 1 <= args.frame_count <= 32):
        parser.error("require envs 1..64, seconds (0,120], vx (0,1], stop (0,10], positive deadline/intervals, nonnegative pushes, frames 1..32")
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic()+args.wall_seconds
    results = []
    errors = []
    for checkpoint_index, path in enumerate(args.checkpoint):
        if time.monotonic() >= deadline:
            break
        args.current_checkpoint = str(Path(path).resolve())
        policy, checkpoint = load_policy(path)
        for case in args.case:
            if time.monotonic() >= deadline:
                break
            folder = args.out/f"{checkpoint_index:02d}_{Path(path).parent.name}_{Path(path).stem}"/case
            try:
                result = evaluate(policy, checkpoint, args, case, folder, deadline)
                results.append(result)
            except Exception as exc:
                errors.append({"checkpoint": args.current_checkpoint, "case": case, "error": repr(exc)})
                print(f"Error {path}/{case}: {exc!r}", flush=True)
                raise
            finally:
                (args.out/"summary.json").write_text(json.dumps({"results": results, "errors": errors,
                    "wall_deadline_reached": time.monotonic() >= deadline}, indent=2, allow_nan=False)+"\n")
    if not results:
        raise SystemExit("No cases completed within wall budget")


if __name__ == "__main__":
    main()
