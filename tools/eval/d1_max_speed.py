"""Stage gate for D1 Max forward-speed curriculum, including sustained moving turns."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

from d1_max_omni import PoseRenderer
import matplotlib.pyplot as plt
import numpy as np
import torch

from rl01.d1_max_omni import OmniEnv
from rl01.train_d1_max_omni import load_policy


def evaluate(checkpoint, output, *, preview_speed=None, turn_rate=.26, seed=2406, render=True):
    actor, cfg = load_policy(checkpoint)
    speed = cfg.max_speed if preview_speed is None else preview_speed
    if preview_speed is not None:
        cfg = replace(cfg, max_speed=speed, wheel_speed_limit=max(15., (speed + 1.05) / .096))
    torch.set_num_threads(1)
    commands = np.array([0., turn_rate, -turn_rate, 0., turn_rate, -turn_rate])
    env = OmniEnv(len(commands), replace(cfg, episode_s=10.), seed=seed, nthread=3, training=False)
    env.randomize_models(np.arange(3, 6), scale=cfg.robustness)
    output.mkdir(parents=True, exist_ok=False)
    steps = 390
    poses = np.empty((steps + 1, len(commands), env.state.shape[1]))
    poses[0] = env.state
    trace = np.zeros((steps, len(commands), 5))
    clearance = np.zeros((steps, len(commands), 4))
    try:
        with torch.inference_mode():
            for i in range(steps):
                t = i * env.dt
                env.command[:, 0] = speed * np.clip((t - .5) / 2.5, 0, 1)
                env.command[:, 2] = commands * np.clip((t - 3.10) / .50, 0, 1)
                if i == 210:
                    env.push(np.arange(3, 6), scale=cfg.robustness, azimuth=np.array([0., np.pi / 2, -np.pi / 2]))
                action = actor(torch.from_numpy(env.observations()[0])).numpy()
                _, _, info = env.step(action)
                trace[i] = np.column_stack([info[k] for k in ("vx", "vy", "yaw", "tilt", "fall")])
                clearance[i] = info["clearance"]
                poses[i + 1] = env.state
        steady = trace[235:]
        by_case = {}
        for i, name in enumerate(("straight", "left_arc", "right_arc", "pushed_straight", "pushed_left_arc", "pushed_right_arc")):
            vx, vy, yaw, tilt, _ = steady[:, i].T
            falls = int(trace[:, i, 4].sum())
            mean_speed = float(vx.mean())
            yaw_rmse = float(np.sqrt(np.mean((yaw - commands[i])**2)))
            mean_yaw = float(yaw.mean())
            peak_tilt = float(np.rad2deg(trace[:, i, 3]).max())
            side_rms = float(np.sqrt(np.mean(vy**2)))
            metrics = {}
            passes = (falls == 0 and abs(mean_speed - speed) <= .18 * speed and peak_tilt <= 15
                      and side_rms <= .25 and yaw_rmse <= (.12 if commands[i] == 0 else .32)
                      and (commands[i] == 0 or .60 <= mean_yaw / commands[i] <= 1.40))
            if commands[i] == 0:
                root = env.root
                w, x, y, z = poses[:, i, root + 3:root + 7].T
                heading = np.unwrap(np.arctan2(2*(w*z + x*y), 1-2*(y*y + z*z)))
                rotation = np.array([[np.cos(heading[0]), -np.sin(heading[0])],
                                     [np.sin(heading[0]), np.cos(heading[0])]])
                xy = poses[:, i, root:root + 2]
                displacement = (xy - xy[0]) @ rotation
                delta = np.rad2deg(heading - heading[0])
                heading_limit = 2.0 if i == 0 else 5.0
                perpendicular_limit = .15 if i == 0 else .40
                max_heading = float(np.max(np.abs(delta)))
                cross_track = float(np.max(np.abs(displacement[:, 1])))
                passes = passes and max_heading <= heading_limit and cross_track <= perpendicular_limit
                metrics = dict(final_heading_drift_deg=float(delta[-1]), peak_heading_drift_deg=max_heading,
                               final_cross_track_m=float(displacement[-1, 1]), peak_cross_track_m=cross_track,
                               forward_distance_m=float(displacement[-1, 0]), heading_limit_deg=heading_limit,
                               cross_track_limit_m=perpendicular_limit)
            by_case[name] = dict(passed=bool(passes), mean_vx=mean_speed, mean_yaw=mean_yaw,
                                 yaw_rmse=yaw_rmse, side_rms=side_rms, peak_tilt_degrees=peak_tilt, falls=falls,
                                 peak_clearance_mm=float(clearance[:, i].max() * 1000), **metrics)
            print(f"speed_case={name} requested_vx={speed:g} actual_vx={mean_speed:.3f} "
                  f"yaw={mean_yaw:+.3f} yaw_rmse={yaw_rmse:.3f} vy_rms={side_rms:.3f} "
                  f"tilt={peak_tilt:.2f} falls={falls} pass={bool(passes)}", flush=True)
        result = {"checkpoint": str(checkpoint), "speed_mps": speed, "yaw_commands": commands.tolist(), "seed": seed,
                  "preview_untrained_speed": preview_speed is not None, "passed": all(c["passed"] for c in by_case.values()),
                  "cases": by_case}
        (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        t = np.arange(steps) * env.dt
        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
        for i in range(3):
            label = list(by_case)[i]
            axes[0].plot(t, trace[:, i, 0], label=label)
            axes[1].plot(t, trace[:, i, 2], label=label)
            axes[2].plot(t, np.rad2deg(trace[:, i, 3]), label=label)
        axes[0].axhline(speed, color=".3", linestyle="--")
        for ax, unit in zip(axes, ("Vx (m/s)", "Yaw rate (rad/s)", "Tilt (deg)")):
            ax.set_ylabel(unit)
            ax.grid(alpha=.25)
        axes[0].legend()
        axes[2].set_xlabel("Simulated time (s)")
        fig.savefig(output / "curves.png", dpi=130)
        plt.close(fig)
        if render:
            visual = OmniEnv(1, cfg, training=False, visual=True)
            try:
                renderer = PoseRenderer(visual.model)
                try:
                    for i, name in enumerate(("left_arc", "right_arc"), start=1):
                        center = 235 + int(clearance[235:, i].max(axis=1).argmax())
                        renderer.save(output / name, {"state": poses}, i, center, 10, env.dt, name)
                finally:
                    renderer.close()
            finally:
                visual.close()
        return result
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--preview-speed", type=float)
    p.add_argument("--turn-rate", type=float, default=.26, help="Magnitude of both signed moving-turn tests, rad/s")
    p.add_argument("--seed", type=int, default=2406)
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--require-pass", action="store_true")
    a = p.parse_args()
    if a.preview_speed is not None and not 0 < a.preview_speed <= 3.5:
        p.error("preview speed must be in (0, 3.5]")
    if not 0 < a.turn_rate <= .55:
        p.error("turn rate must be in (0, 0.55]")
    result = evaluate(a.checkpoint, a.out, preview_speed=a.preview_speed, turn_rate=a.turn_rate,
                      seed=a.seed, render=not a.no_render)
    return int(a.require_pass and not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
