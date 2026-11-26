"""Evaluate the real WebLab SimEngine without opening an HTTP server (MuJoCo only).

Run sequentially after pausing CPU-heavy training::
    uv run python tools/eval/web_controls.py --checkpoint runs/<run>/model_300.pt \
        --seconds 3 --repetitions 1 --wall-seconds 120 --out runs/web_controls.json

Commands share web/server.py's mapping, including backward scaling, diagonal
normalization and speed-matched curvature. --speed-limit defaults to 1.2 m/s.
Extra WQ/WE cases test --turn-speeds (default 0.3/0.6/1.2 m/s). Every independent
case starts with the real standby/reset flow; move-stop-move never resets between
its three phases. Existing automatic fall resets remain enabled and are reported.
No contact, actuator, observation, or model parameters are replaced with training
settings. Deterministic policy actions, one environment and one Torch CPU thread.
JSON is saved after every phase, including a partial phase on budget expiry.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
CASES = ("W", "S", "A", "D", "Q", "E", "WA", "WD", "SA", "SD", "WQ", "WE")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", type=Path, help="JSON path; defaults beside checkpoint")
    parser.add_argument("--seconds", type=float, default=3.0, help="simulated seconds per command phase")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--wall-seconds", type=float, default=120.0)
    parser.add_argument("--speed-limit", type=float, default=1.2, help="same setting as webpage speed limit")
    parser.add_argument("--turn-radius", type=float, default=1.0, help="moving turn radius in metres [0.25, 5]")
    parser.add_argument("--inplace-yaw-rate", type=float, default=0.8, help="stationary yaw rate in rad/s [0.3, 1.2]")
    parser.add_argument("--turn-speeds", nargs="*", type=float, default=[0.3, 0.6, 1.2],
                        help="additional forward+left/right turn speeds; omit values to disable extra cases")
    parser.add_argument("--settle-seconds", type=float, default=1.0, help="standby settling per independent case")
    args = parser.parse_args()
    for name in ("seconds", "wall_seconds", "speed_limit"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not math.isfinite(args.settle_seconds) or args.settle_seconds < 0:
        parser.error("--settle-seconds must be finite and nonnegative")
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    if not 0.2 <= args.speed_limit <= 2.5:
        parser.error("--speed-limit must be within the webpage's [0.2, 2.5] range")
    if not math.isfinite(args.turn_radius) or not 0.25 <= args.turn_radius <= 5:
        parser.error("--turn-radius must be within [0.25, 5]")
    if not math.isfinite(args.inplace_yaw_rate) or not 0.3 <= args.inplace_yaw_rate <= 1.2:
        parser.error("--inplace-yaw-rate must be within [0.3, 1.2]")
    if any(not math.isfinite(speed) or not 0.2 <= speed <= 2.5 for speed in args.turn_speeds):
        parser.error("--turn-speeds values must be within [0.2, 2.5]")
    args.checkpoint = args.checkpoint.resolve()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    args.out = (args.out or args.checkpoint.with_name(f"web_controls_{args.checkpoint.stem}.json")).resolve()
    if args.out == args.checkpoint:
        parser.error("--out must differ from --checkpoint")
    return args


def distribution(values):
    values = [value for value in values if value is not None and math.isfinite(value)]
    if not values:
        return {"n": 0, "mean": None, "rms": None, "min": None, "max": None}
    n = len(values)
    return {"n": n, "mean": sum(values) / n,
            "rms": math.sqrt(sum(x * x for x in values) / n),
            "min": min(values), "max": max(values)}


def sample_state(engine, quat_to_rot_and_euler):
    rotation, roll, pitch, yaw = quat_to_rot_and_euler(engine.data.qpos[3:7])
    velocity = rotation.T @ engine.data.qvel[:3]
    height = float(engine.data.qpos[2])
    cosine = float(rotation[2, 2])
    heading_error = math.atan2(math.sin(engine.target_yaw - yaw), math.cos(engine.target_yaw - yaw))
    speed = math.hypot(float(velocity[0]), float(velocity[1]))
    return {"time": float(engine.data.time), "position": engine.data.qpos[:3].tolist(),
            "vx": float(velocity[0]), "vy": float(velocity[1]), "wz": float(engine.data.qvel[5]),
            "height": height, "roll_deg": math.degrees(roll), "pitch_deg": math.degrees(pitch),
            "tilt_deg": math.degrees(math.acos(max(-1.0, min(1.0, cosine)))),
            "yaw_deg": math.degrees(yaw), "target_yaw_deg": math.degrees(engine.target_yaw),
            "heading_error_deg": math.degrees(heading_error),
            "curvature_per_m": float(engine.data.qvel[5]) / speed if speed > 0.05 else None,
            "fallen": height < 0.14 or cosine < 0.5}


def run_phase(engine, quat_to_rot_and_euler, name, command, seconds, deadline):
    engine.set_command(*command)  # one browser command event; auto-reset semantics stay unchanged
    requested_steps = math.ceil(seconds / engine.dt)
    start = sample_state(engine, quat_to_rot_and_euler)
    previous = start
    samples = []
    falls_before = engine.falls
    auto_resets = 0
    yaw_delta = 0.0
    for _ in range(requested_steps):
        if time.perf_counter() >= deadline:
            break
        engine.step_once()
        current = sample_state(engine, quat_to_rot_and_euler)
        reset = current["time"] < previous["time"]
        auto_resets += int(reset)
        if not reset:
            dyaw = math.radians(current["yaw_deg"] - previous["yaw_deg"])
            yaw_delta += math.degrees(math.atan2(math.sin(dyaw), math.cos(dyaw)))
        samples.append(current)
        previous = current
    metrics = ("vx", "vy", "wz", "height", "tilt_deg", "roll_deg", "pitch_deg", "heading_error_deg", "curvature_per_m")
    last_half = samples[len(samples) // 2:]
    result = {
        "phase": name, "mode": engine.mode, "command_body": command,
        "command_curvature_per_m": command[2] / math.hypot(*command[:2]) if math.hypot(*command[:2]) > 1e-6 else None,
        "requested_seconds": seconds, "simulated_seconds": len(samples) * engine.dt,
        "complete": len(samples) == requested_steps,
        "steps": len(samples), "falls": int(engine.falls - falls_before),
        "auto_resets": auto_resets,
        "fallen_frames": sum(int(sample["fallen"]) for sample in samples),
        "start": start, "end": previous,
        "yaw_delta_deg_excluding_resets": yaw_delta,
        "displacement_world": [previous["position"][i] - start["position"][i] for i in range(3)],
        "all_samples": {key: distribution([s[key] for s in samples]) for key in metrics},
        "last_half": {key: distribution([s[key] for s in last_half]) for key in metrics},
    }
    if auto_resets:
        result["note"] = "Automatic reset occurred: displacement includes resets and commanded motion may have been zeroed by the webpage engine."
    return result


def save_report(report, output, started):
    report["wall_elapsed_seconds"] = time.perf_counter() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)


def main():
    args = parse_args()  # --help remains a lightweight check without importing simulation libraries
    started = time.perf_counter()
    deadline = started + args.wall_seconds
    report = {
        "checkpoint": str(args.checkpoint), "verification_layer": "MuJoCo WebLab original scene only",
        "status": "running", "num_envs": 1, "torch_threads": 1, "deterministic": True,
        "settings": {"seconds_per_phase": args.seconds, "repetitions": args.repetitions,
                     "speed_limit": args.speed_limit, "settle_seconds": args.settle_seconds,
                     "turn_radius": args.turn_radius, "inplace_yaw_rate": args.inplace_yaw_rate,
                     "turn_speeds": args.turn_speeds,
                     "wall_seconds": args.wall_seconds},
        "conditions": "Real web.server.SimEngine; original scene contact settings; no added noise, randomization, pushes or payload; auto fall resets retained.",
        "cases": [],
    }
    engine = None
    save_report(report, args.out, started)
    try:
        import torch
        torch.set_num_threads(1)
        spec = importlib.util.spec_from_file_location("rl01_web_server_eval", ROOT / "web" / "server.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        engine = module.SimEngine()
        loaded = engine.load_checkpoint(str(args.checkpoint))
        if not loaded["ok"]:
            raise RuntimeError(f"checkpoint load failed: {loaded.get('error')}")
        engine.speed_limit = args.speed_limit
        engine.set_turn_config(args.turn_radius, args.inplace_yaw_rate)
        report["policy_info"] = engine.policy_info
        report["obs_taps"] = list(engine.obs_taps)
        cases = [(name, name, args.speed_limit) for name in CASES]
        cases.extend((f"{keys}@{speed:g}", keys, speed) for speed in args.turn_speeds for keys in ("WQ", "WE"))
        cases.append(("move-stop-move", "W", args.speed_limit))
        for repetition in range(args.repetitions):
            for case_name, case_keys, case_speed in cases:
                if time.perf_counter() >= deadline:
                    report["status"] = "deadline"
                    return 2
                engine.set_mode("standby")
                engine.reset()
                engine.speed_limit = case_speed
                entry = {"case": case_name, "speed_limit": case_speed, "repetition": repetition + 1, "phases": []}
                report["cases"].append(entry)
                preparation = run_phase(engine, module.quat_to_rot_and_euler, "standby_settle",
                                        [0.0, 0.0, 0.0], args.settle_seconds, deadline)
                entry["preparation"] = preparation
                if not preparation["complete"]:
                    report["status"] = "deadline"
                    return 2
                engine.set_mode("rl")
                phases = [(case_name, case_keys)] if case_name != "move-stop-move" else [
                    ("move_1", "W"), ("stop", ""), ("move_2", "W")]
                for phase_name, keys in phases:
                    command = module.keyboard_command(keys, case_speed, args.turn_radius, args.inplace_yaw_rate)
                    phase = run_phase(engine, module.quat_to_rot_and_euler, phase_name,
                                      command, args.seconds, deadline)
                    entry["phases"].append(phase)
                    save_report(report, args.out, started)
                    mean = phase["last_half"]
                    print(f"[web-controls] rep={repetition + 1} case={case_name} phase={phase_name} "
                          f"steps={phase['steps']} vx={mean['vx']['mean']} vy={mean['vy']['mean']} "
                          f"wz={mean['wz']['mean']} falls={phase['falls']} resets={phase['auto_resets']}", flush=True)
                    if not phase["complete"]:
                        report["status"] = "deadline"
                        return 2
        report["status"] = "complete"
        return 0
    except Exception as exc:
        report["status"] = "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if engine is not None:
            engine.set_command()
            engine.running = False
        save_report(report, args.out, started)
        print(f"[web-controls] status={report['status']} JSON={args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
