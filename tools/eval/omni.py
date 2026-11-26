"""Independent flat-ground omni/heading/restart evaluation (MuJoCo only).

Run baseline and candidate sequentially, while no CPU-heavy training is running::
    MUJOCO_GL=egl uv run python tools/eval/omni.py \
        runs/go2_vel2_v6b/model_300.pt runs/new/model_300.pt \
        --profile stress --seconds 24 --num-envs 8 --out runs/omni_compare
    # Repeat with --sampled; use identical seed/profile/seconds for comparison.

--seconds is simulated time PER scenario; --wall-seconds is the entire script's
wall-clock budget. Completed cases are saved incrementally, including on timeout.
Tracking is body-frame; path/heading errors use the initial yaw integrated with the
commanded yaw rate, so accumulated heading drift cannot disappear into body axes.
Only the first episode of each environment contributes. Terminal states come from
state_buf, before the environment's automatic reset. No hardware is accessed.
"""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

# Must precede MuJoCo import; caller can explicitly select another GL backend.
os.environ.setdefault("MUJOCO_GL", "egl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw
import torch

from rl01.env import EnvCfg, Go2Env, S_ANG, S_LIN, S_QUAT, build_train_model, quat_to_R
from rl01.play import load_policy


class BudgetExpired(RuntimeError):
    pass


def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def summary(x):
    """JSON-safe distribution; missing samples remain null, never a false zero."""
    v = np.asarray(x, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if not v.size:
        return {"n": 0, "mean": None, "rms": None, "p95_abs": None, "max_abs": None}
    return {"n": int(v.size), "mean": float(v.mean()),
            "rms": float(np.sqrt(np.mean(v * v))),
            "p95_abs": float(np.percentile(np.abs(v), 95)), "max_abs": float(np.abs(v).max())}


def command(name, t, seconds, speed=0.7, yaw_rate=0.6, turn_radius=1.0):
    regression_speeds = {"forward_slow": 0.3, "forward_fast": 1.5, "forward_max": 2.0}
    if name in regression_speeds:
        return np.array([regression_speeds[name], 0., 0.])
    directions = {"forward": (1, 0), "backward": (-1, 0),
                  "left": (0, 1), "right": (0, -1),
                  "forward_left": (1, 1), "forward_right": (1, -1),
                  "backward_left": (-1, 1), "backward_right": (-1, -1)}
    if name in directions:
        v = np.asarray(directions[name], float)
        return np.array([*(speed * v / np.linalg.norm(v)), 0.])
    if name == "rotate_left":
        return np.array([0., 0., yaw_rate])
    if name == "rotate_right":
        return np.array([0., 0., -yaw_rate])
    if name in ("arc_left", "arc_right"):
        direction = 1. if name == "arc_left" else -1.
        return np.array([speed, 0., direction * min(yaw_rate, speed / turn_radius)])
    if name == "long_forward":
        return np.array([1., 0., 0.])
    if name == "restart":
        # Four move bouts separated by three complete stand intervals.
        phase = min(6, int(t / (seconds / 7)))
        return np.array([speed if phase % 2 == 0 else 0., 0., 0.])
    if name == "direction_switch":
        cmds = [(speed, 0, 0), (0, 0, 0), (0, speed, 0), (0, 0, 0),
                (-speed, 0, 0), (0, 0, 0), (0, -speed, 0)]
        return np.array(cmds[min(6, int(t / (seconds / 7)))], float)
    raise ValueError(name)


SCENARIOS = ["forward", "backward", "left", "right", "forward_left", "forward_right",
             "backward_left", "backward_right", "rotate_left", "rotate_right",
             "arc_left", "arc_right", "forward_slow", "forward_fast", "forward_max",
             "long_forward", "restart", "direction_switch"]


def make_cfg(ck, profile, seconds):
    # Checkpoint is authoritative, including optional heading obs and history taps.
    cfg = EnvCfg(**ck["env_cfg"])
    for key in ("friction_range", "mass_range", "kp_range", "kd_range", "damping_range", "obs_taps"):
        setattr(cfg, key, tuple(getattr(cfg, key)))
    cfg.stairs = False
    cfg.flat_frac = 0.
    # Fixed evaluation commands replace the training command curriculum. Using
    # one reset sampler also avoids old/new omni branches consuming different
    # numbers of random draws before the first commanded step.
    cfg.omni = False
    cfg.flat_cmd_step_prob = 0.0
    cfg.episode_s = seconds + 2.
    cfg.cmd_slew_rate = 0.  # This evaluator supplies the actual command, including step transitions.
    # Clean/stress are fixed comparisons; training retains the checkpoint's
    # attained disturbance/DR strength, not the curriculum's original start.
    if profile in ("clean", "stress"):
        cfg.robustness_curriculum = False
    elif cfg.robustness_curriculum:
        cfg.robustness_init = float(ck.get("robustness_scale", cfg.robustness_init))
    if profile == "clean":
        cfg.noise = False
        cfg.push_vel = cfg.push_z = cfg.push_ang = 0.
        cfg.friction_range = (0.8, 0.8)
        cfg.mass_range = (0., 0.)
        cfg.kp_range = cfg.kd_range = cfg.damping_range = (1., 1.)
        cfg.com_range = 0.
    elif profile == "stress":
        # Common conditions for fair old/new comparisons, inspired by stairs v9.
        cfg.noise = True
        cfg.friction_range = (0.35, 1.25)
        cfg.mass_range = (-1.2, 2.)
        cfg.kp_range = cfg.kd_range = (0.82, 1.18)
        cfg.damping_range = (0.45, 1.25)
        cfg.com_range = 0.035
        cfg.push_vel, cfg.push_z, cfg.push_ang = 0.6, 0.2, 0.45
        cfg.push_interval_s = 4.5
    return cfg


def set_command(env, cmd):
    env.cmd[:] = cmd
    env.cmd_target[:] = cmd
    env.cmd_left[:] = 10**9
    env._compute_obs(shift=False)


def film_schedule(name, seconds, dt):
    if name in ("restart", "direction_switch"):
        # Consecutive frames around first stop and first restart, not sparse poses.
        anchors = [seconds / 7 - 0.16, seconds * 2 / 7 - 0.16]
    else:
        anchors = [min(2., seconds / 3)]
    return {max(0, round((t + j * 0.08) / dt)) for t in anchors for j in range(8)
            if 0 <= t + j * 0.08 < seconds}


def save_filmstrip(frames, path):
    if not frames:
        return None
    width, height = frames[0][1].shape[1], frames[0][1].shape[0]
    columns = 4
    canvas = Image.new("RGB", (columns * width, math.ceil(len(frames) / columns) * (height + 28)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, pixels) in enumerate(frames):
        x, y = (i % columns) * width, (i // columns) * (height + 28)
        canvas.paste(Image.fromarray(pixels), (x, y + 28))
        draw.text((x + 6, y + 6), label, fill="black")
    canvas.save(path)
    return str(path)


def render_state(renderer, model, data, state):
    mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    mujoco.mj_forward(model, data)
    yaw = math.degrees(math.atan2(data.xmat[model.body("base").id, 3], data.xmat[model.body("base").id, 0]))
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [state[1], state[2], max(0.18, state[3])]
    cam.distance, cam.elevation, cam.azimuth = 1.25, -10, yaw + 90
    renderer.update_scene(data, camera=cam)
    return renderer.render().copy()


def finite_mean(x, axis):
    """Mean of available samples, preserving NaN when the entire slice failed."""
    finite = np.isfinite(x)
    count = finite.sum(axis=axis)
    total = np.where(finite, x, 0.).sum(axis=axis)
    return np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)


def trajectory_plot(trace, path, title):
    t, xy, desired = trace["t"], trace["xy"], trace["desired_xy"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for i in range(xy.shape[1]):
        axes[0].plot(xy[:, i, 0], xy[:, i, 1], alpha=0.6, lw=1)
        axes[0].plot(desired[:, i, 0], desired[:, i, 1], "k--", alpha=0.2, lw=0.8)
    axes[0].set(xlabel="initial-heading x (m)", ylabel="initial-heading y (m)", title="Actual paths / dashed command paths")
    axes[0].axis("equal")
    for k, label in enumerate(("vx", "vy", "wz")):
        axes[1].plot(t, finite_mean(trace["velocity"][:, :, k], axis=1), label=label)
        axes[1].plot(t, trace["command"][:, k], "--", alpha=0.6)
    axes[1].set(xlabel="time (s)", ylabel="m/s or rad/s", title="Body velocity / dashed commands")
    axes[1].legend()
    axes[2].plot(t, finite_mean(np.linalg.norm(xy - desired, axis=2), axis=1), label="position error m")
    axes[2].plot(t, finite_mean(np.abs(trace["heading_error"]), axis=1), label="heading error rad")
    axes[2].set(xlabel="time (s)", title="Accumulated drift (surviving samples)")
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=135)
    plt.close(fig)


def segment_metrics(trace, dt):
    cmds, ts, xy = trace["command"], trace["t"], trace["xy"]
    changes = np.flatnonzero(np.any(np.diff(cmds, axis=0) != 0, axis=1)) + 1
    # Also split straight trajectories every five seconds to expose growing drift.
    cuts = np.unique(np.r_[0, changes, np.arange(round(5 / dt), len(ts), round(5 / dt)), len(ts)])
    segments = []
    for start, end in zip(cuts[:-1], cuts[1:]):
        c = cmds[start]
        moving = np.linalg.norm(c[:2]) > 0.05
        phase_start = max([0, *changes[changes <= start].tolist()])
        phase_age = ts[start:end] - ts[phase_start]
        settled = phase_age >= min(1., max(dt, (ts[end - 1] - ts[start]) / 2))
        err = trace["velocity"][start:end] - cmds[start:end, None, :]
        delta = xy[end - 1] - xy[start]
        desired_delta = trace["desired_xy"][end - 1] - trace["desired_xy"][start]
        initial_yaw = trace["reference_yaw"][start]
        direction = np.stack([np.cos(initial_yaw) * c[0] - np.sin(initial_yaw) * c[1],
                              np.sin(initial_yaw) * c[0] + np.cos(initial_yaw) * c[1]], axis=1)
        direction /= max(np.linalg.norm(c[:2]), 1e-9)
        # xy lives in the initial-heading frame; reference_yaw uses that same frame.
        cross = (delta - desired_delta)[:, 0] * -direction[:, 1] + (delta - desired_delta)[:, 1] * direction[:, 0]
        segments.append({"start_s": float(ts[start] - dt), "end_s": float(ts[end - 1]),
                         "command": c.tolist(), "moving": bool(moving),
                         "velocity_xy_error_mps": summary(np.linalg.norm(err[settled, :, :2], axis=2)),
                         "yaw_rate_error_radps": summary(err[settled, :, 2]),
                         "segment_position_error_m": summary(np.linalg.norm(delta - desired_delta, axis=1)),
                         "segment_cross_track_m": summary(cross) if moving else None,
                         "heading_error_deg": summary(np.degrees(trace["heading_error"][start:end])),
                         "stop_speed_mps": summary(np.linalg.norm(trace["velocity"][start:end][settled, :, :2], axis=2)) if not moving and abs(c[2]) < .05 else None})
    transitions = []
    for start in changes:
        prev, target = cmds[start - 1], cmds[start]
        if np.linalg.norm(prev[:2]) <= 0.05 < np.linalg.norm(target[:2]):
            kind, tolerance = "restart", max(0.12, 0.2 * np.linalg.norm(target[:2]))
        elif np.linalg.norm(target[:2]) <= .05 < np.linalg.norm(prev[:2]):
            kind, tolerance = "stop", .1
        else:
            continue
        next_changes = changes[changes > start]
        end = int(next_changes[0]) if len(next_changes) else len(ts)
        errors = np.linalg.norm(trace["velocity"][start:end, :, :2] - target[:2], axis=2)
        # A response must remain within tolerance for 0.3 s. Failed/terminated
        # environments retain null latency and count as unsuccessful.
        hold = max(1, round(.3 / dt))
        latency = np.full(errors.shape[1], np.nan)
        for j in range(max(0, len(errors) - hold + 1)):
            hit = np.all(errors[j:j + hold] <= tolerance, axis=0) & ~np.isfinite(latency)
            latency[hit] = j * dt
        transitions.append({"kind": kind, "time_s": float(ts[start] - dt),
                            "target": target.tolist(), "tolerance_mps": float(tolerance),
                            "hold_s": hold * dt, "successful_envs": int(np.isfinite(latency).sum()),
                            "latency_s": summary(latency),
                            "first_second_error_mps": summary(errors[:round(1 / dt)]),
                            "settled_error_mps": summary(errors[round(1 / dt):]),
                            "per_env_latency_s": [float(x) if np.isfinite(x) else None for x in latency]})
    return segments, transitions


def foot_metrics(trace, steady, dt):
    """Foot-centre geometry on flat ground; contact is a height proxy only.

    Copying sensor samples into trace is required: rollout overwrites its buffers.
    Liftoffs use hysteresis (>3.5 cm up, <=3.0 cm down); the initial drop and
    samples after a first terminal state never count as alternating footsteps.
    """
    heights = trace["foot_height"][steady]
    valid = np.isfinite(heights)
    lifted = heights > .035
    contact = heights <= .03
    per_leg = {}
    for leg, name in enumerate(("FL", "FR", "RL", "RR")):
        z = heights[:, :, leg]
        available = valid[:, :, leg]
        finite_z = z[available]
        per_leg[name] = {
            "height_m": summary(z),
            "height_p95_m": float(np.percentile(finite_z, 95)) if finite_z.size else None,
            "swing_height_m": summary(z[lifted[:, :, leg]]),
            "swing_fraction": float(lifted[:, :, leg][available].mean()) if available.any() else None,
            "contact_height_proxy_fraction": float(contact[:, :, leg][available].mean()) if available.any() else None,
        }
    available = valid.all(axis=2)
    all_down = ~lifted.any(axis=2)
    result = {"lift_threshold_m": .035, "contact_proxy_threshold_m": .03,
              "contact_measurement": "foot-centre height proxy, not measured touch/force",
              "per_leg": per_leg,
              "all_four_not_lifted_fraction": float(all_down[available].mean()) if available.any() else None,
              "any_foot_lifted_fraction": float((~all_down)[available].mean()) if available.any() else None,
              "per_env_liftoff_events": []}
    alternating_counts, diagonal_counts, event_rates = [], [], []
    indices = np.flatnonzero(steady)
    for env_index in range(heights.shape[1]):
        events = []
        in_air = np.zeros(4, bool)
        initialized = False
        for index in indices:
            z = trace["foot_height"][index, env_index]
            if not np.isfinite(z).all():
                continue
            if not initialized:
                in_air = z > .035
                initialized = True
                continue
            next_air = np.where(z > .035, True, np.where(z <= .03, False, in_air))
            ups = np.flatnonzero(next_air & ~in_air)
            in_air = next_air
            if not len(ups):
                continue
            mask = sum(1 << int(leg) for leg in ups)
            t = float(trace["t"][index])
            # Cluster a diagonal pair staggered by <=60 ms into one footstep.
            if events and t - events[-1]["time_s"] <= .06:
                events[-1]["mask"] |= mask
            else:
                events.append({"time_s": t, "mask": mask})
        alternate = diagonal_alternate = 0
        previous_mask = previous_diagonal = 0
        for event in events:
            mask = event.pop("mask")
            event["legs"] = [name for leg, name in enumerate(("FL", "FR", "RL", "RR")) if mask & (1 << leg)]
            event["alternates_previous"] = bool(previous_mask and not (mask & previous_mask))
            alternate += int(event["alternates_previous"])
            diagonal = 1 if mask & 0b1001 and not mask & 0b0110 else (2 if mask & 0b0110 and not mask & 0b1001 else 0)
            event["diagonal_group"] = diagonal or None
            diagonal_alternate += int(bool(previous_diagonal and diagonal and previous_diagonal != diagonal))
            previous_mask, previous_diagonal = mask, diagonal
        duration = float(available[:, env_index].sum() * dt)
        if duration:
            alternating_counts.append(alternate)
            diagonal_counts.append(diagonal_alternate)
            event_rates.append(len(events) / duration)
        result["per_env_liftoff_events"].append({"env": env_index, "valid_duration_s": duration,
                                                "events": events, "alternating_events": alternate if duration else None,
                                                "alternating_diagonal_events": diagonal_alternate if duration else None})
    result["alternating_liftoff_events"] = summary(alternating_counts)
    result["alternating_diagonal_events"] = summary(diagonal_counts)
    result["liftoff_events_per_second"] = summary(event_rates)
    # A low swinging foot can move fast without being in contact. Restrict this
    # conservative drag proxy to low feet with nearly zero vertical velocity,
    # and require both endpoints to qualify. It remains geometry, not force.
    xyz = trace["foot_xyz"]
    foot_velocity = np.diff(xyz, axis=0) / dt
    finite_pairs = np.isfinite(xyz[:-1]).all(axis=3) & np.isfinite(xyz[1:]).all(axis=3)
    finite_pairs &= (steady[:-1] & steady[1:])[:, None, None]
    eligible = finite_pairs & (xyz[:-1, :, :, 2] <= .028) & (xyz[1:, :, :, 2] <= .028)
    eligible &= np.abs(foot_velocity[:, :, :, 2]) < .05
    horizontal_speed = np.linalg.norm(foot_velocity[:, :, :, :2], axis=3)
    def drag_summary(selected, available):
        v = horizontal_speed[selected]
        return {"n_samples": int(v.size), "mean_mps": float(v.mean()) if v.size else None,
                "p95_mps": float(np.percentile(v, 95)) if v.size else None,
                "eligible_fraction_of_finite_pairs": float(selected.sum() / available.sum()) if available.any() else None}
    result["support_drag_proxy"] = {
        "measurement": "world-foot finite difference; height/low-vz proxy, not force contact or proven slip",
        "max_foot_centre_z_m": .028, "max_abs_vertical_velocity_mps": .05,
        "both_endpoints_below_height": True,
        **drag_summary(eligible, finite_pairs),
    }
    for leg, name in enumerate(("FL", "FR", "RL", "RR")):
        selected = np.zeros_like(eligible)
        available = np.zeros_like(finite_pairs)
        selected[:, :, leg] = eligible[:, :, leg]
        available[:, :, leg] = finite_pairs[:, :, leg]
        per_leg[name]["support_drag_proxy"] = drag_summary(selected, available)
    return result


def evaluate(ac, ck, args, profile, scenario, folder, deadline):
    cfg = make_cfg(ck, profile, args.seconds)
    env = Go2Env(args.num_envs, cfg, seed=args.seed, nthread=2)
    renderer = None
    frames = []
    want_frames = args.frames == "all" or (args.frames == "key" and scenario in ("long_forward", "restart", "direction_switch"))
    try:
        if want_frames:
            model = build_train_model(visual=True)
            model.light_castshadow[:] = False
            data = mujoco.MjData(model)
            renderer = mujoco.Renderer(model, 240, 360)
        frame_steps = film_schedule(scenario, args.seconds, env.dt)
        env._compute_obs(shift=False)
        initial_yaw = np.arctan2(env.R[:, 1, 0], env.R[:, 0, 0])
        yaw_ref = initial_yaw.copy()
        initial_xy = env.state[:, 1:3].copy()
        desired_xy = initial_xy.copy()
        active = np.ones(env.N, bool)
        first_fall_time = np.full(env.N, np.nan)
        push_angles = []
        trace = {k: [] for k in ("t", "xy", "desired_xy", "velocity", "command", "heading_error", "reference_yaw", "height", "roll", "pitch", "ang_xy", "active", "foot_height", "foot_xyz")}
        nsteps = max(1, round(args.seconds / env.dt))
        last_progress = time.monotonic()
        for step in range(nsteps):
            now = time.monotonic()
            if now >= deadline:
                raise BudgetExpired(f"wall deadline during {profile}/{scenario}, step {step}/{nsteps}")
            cmd = command(scenario, step * env.dt, args.seconds, args.speed, args.yaw_rate, args.turn_radius)
            set_command(env, cmd)
            obs, _ = env.get_obs()
            if obs.shape[1] != ac.actor[0].in_features:
                raise ValueError(f"actor/environment obs mismatch: {ac.actor[0].in_features} vs {obs.shape[1]}")
            with torch.inference_mode():
                o = torch.as_tensor(obs)
                action = (ac.dist(o).sample() if args.sampled else ac.actor(o)).numpy()
            _, done, _ = env.step(action)
            raw = env.state_buf[:, -1].copy()
            rot = quat_to_R(raw[:, S_QUAT])
            velocity = np.c_[np.einsum("nji,nj->ni", rot, raw[:, S_LIN])[:, :2], raw[:, 25]]
            desired_xy += env.dt * np.stack([np.cos(yaw_ref) * cmd[0] - np.sin(yaw_ref) * cmd[1],
                                            np.sin(yaw_ref) * cmd[0] + np.cos(yaw_ref) * cmd[1]], axis=1)
            yaw_ref += cmd[2] * env.dt
            yaw = np.arctan2(rot[:, 1, 0], rot[:, 0, 0])
            fall = env.last_outcome[1]
            first_fall_time[active & fall] = (step + 1) * env.dt
            # Include a terminal physical sample, then mask every reset sample.
            valid = active.copy()
            def mask(v):
                v = np.array(v, dtype=float, copy=True)
                v[~valid] = np.nan
                return v
            def local(v):
                d = v - initial_xy
                return np.stack([np.cos(initial_yaw) * d[:, 0] + np.sin(initial_yaw) * d[:, 1],
                                 -np.sin(initial_yaw) * d[:, 0] + np.cos(initial_yaw) * d[:, 1]], axis=1)
            values = {"t": (step + 1) * env.dt, "xy": mask(local(raw[:, 1:3])),
                      "desired_xy": mask(local(desired_xy)), "velocity": mask(velocity),
                      "command": cmd.copy(), "heading_error": mask(wrap(yaw - yaw_ref)),
                      "reference_yaw": mask(wrap(yaw_ref - initial_yaw)), "height": mask(raw[:, 3]),
                      "roll": mask(np.arctan2(rot[:, 2, 1], rot[:, 2, 2])),
                      "pitch": mask(np.arcsin(np.clip(-rot[:, 2, 0], -1., 1.))),
                      "ang_xy": mask(raw[:, S_ANG][:, :2]), "active": valid,
                      "foot_height": mask(env.sens_buf[:, -1, 2:12:3]),
                      "foot_xyz": mask(env.sens_buf[:, -1, :12].reshape(env.N, 4, 3))}
            for key, value in values.items():
                trace[key].append(value)
            pushed = np.linalg.norm(env.push_delta, axis=1) > 0
            push_angles.extend(np.arctan2(env.push_delta[valid & pushed, 1], env.push_delta[valid & pushed, 0]).tolist())
            if renderer is not None and step in frame_steps and valid[0]:
                frames.append((f"t={(step + 1) * env.dt:.2f}s cmd={cmd[0]:+.1f},{cmd[1]:+.1f},{cmd[2]:+.1f}",
                               render_state(renderer, model, data, raw[0])))
            active &= ~done
            if now - last_progress > 30:
                print(f"progress profile={profile} case={scenario} sim_s={(step + 1) * env.dt:.1f} active={active.sum()}/{env.N}", flush=True)
                last_progress = now
        trace = {k: np.asarray(v) for k, v in trace.items()}
        segments, transitions = segment_metrics(trace, env.dt)
        steady = trace["t"] >= min(2., args.seconds / 4)
        err = trace["velocity"] - trace["command"][:, None, :]
        path_error = np.linalg.norm(trace["xy"] - trace["desired_xy"], axis=2)
        last_indices = np.maximum(0, np.sum(trace["active"], axis=0) - 1)
        last_path_error = path_error[last_indices, np.arange(env.N)]
        metrics = {"scenario": scenario, "seconds": args.seconds, "num_envs": env.N,
                   "survived_envs": int(active.sum()), "first_episode_falls": int(np.isfinite(first_fall_time).sum()),
                   "first_fall_time_s": [float(x) if np.isfinite(x) else None for x in first_fall_time],
                   "completed_fraction": float(trace["active"].mean()),
                   "velocity_xy_error_mps": summary(np.linalg.norm(err[steady, :, :2], axis=2)),
                   "actual_yaw_rate_radps": summary(trace["velocity"][steady, :, 2]),
                   "command_yaw_rate_radps": summary(trace["command"][steady, 2]),
                   "yaw_rate_error_radps": summary(err[steady, :, 2]),
                   "height_m": summary(trace["height"][steady]),
                   "roll_deg": summary(np.degrees(trace["roll"][steady])),
                   "pitch_deg": summary(np.degrees(trace["pitch"][steady])),
                   "angular_xy_radps": summary(trace["ang_xy"][steady]),
                   "heading_error_deg": summary(np.degrees(trace["heading_error"][steady])),
                   "path_error_m": summary(path_error), "last_valid_path_error_m": summary(last_path_error),
                   "push_events_in_first_episodes": len(push_angles),
                   "push_bearing_octants": np.histogram(push_angles, bins=np.linspace(-np.pi, np.pi, 9))[0].tolist(),
                   "segments": segments, "transitions": transitions, "env_cfg": asdict(cfg),
                   "effective_robustness_scale": float(env.robustness_scale),
                   "feet": foot_metrics(trace, steady, env.dt)}
        if scenario in ("rotate_left", "rotate_right", "arc_left", "arc_right"):
            wz = trace["velocity"][steady, :, 2]
            commanded_wz = trace["command"][steady, 2, None]
            finite = np.isfinite(wz)
            metrics["turning"] = {
                "actual_yaw_rate_radps": summary(wz),
                "command_yaw_rate_radps": summary(commanded_wz),
                "yaw_rate_error_radps": summary(wz - commanded_wz),
                "actual_command_yaw_rate_ratio": summary(wz / commanded_wz),
                "correct_direction_fraction": float((wz * np.sign(commanded_wz) > .05)[finite].mean()) if finite.any() else None,
                "meaningful_direction_threshold_radps": .05,
            }
            if scenario in ("rotate_left", "rotate_right"):
                displacement = np.linalg.norm(trace["xy"], axis=2)
                metrics["turning"]["translation_displacement_m"] = summary(displacement[steady])
                metrics["turning"]["last_valid_translation_displacement_m"] = summary(displacement[last_indices, np.arange(env.N)])
                metrics["turning"]["translation_speed_mps"] = summary(np.linalg.norm(trace["velocity"][steady, :, :2], axis=2))
        initial_cmd = trace["command"][0, :2]
        if np.linalg.norm(initial_cmd) > .05 and np.all(trace["command"] == trace["command"][0]):
            normal = np.array([-initial_cmd[1], initial_cmd[0]]) / np.linalg.norm(initial_cmd)
            cross_track = (trace["xy"] - trace["desired_xy"]) @ normal
            metrics["cross_track_m"] = summary(cross_track)
            metrics["last_valid_cross_track_m"] = summary(cross_track[last_indices, np.arange(env.N)])
        stem = folder / scenario
        trajectory_plot(trace, stem.with_suffix(".png"), f"{folder.name}: {scenario}")
        metrics["trajectory_png"] = str(stem.with_suffix(".png"))
        metrics["filmstrip_png"] = save_filmstrip(frames, folder / f"{scenario}_frames.png")
        # Compact raw arrays preserve per-seed evidence and permit offline metrics.
        np.savez_compressed(folder / f"{scenario}_trace.npz", **trace)
        stem.with_suffix(".json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
        return metrics
    finally:
        if renderer is not None:
            renderer.close()
        env.roller.close()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", nargs="+", help="One or more checkpoints; pass baseline then candidate for comparison")
    p.add_argument("--seconds", type=float, default=24., help="Simulated seconds per scenario (default: 24)")
    p.add_argument("--wall-seconds", type=float, default=600., help="Entire run wall-clock deadline (default: 600)")
    p.add_argument("--num-envs", type=int, default=8, choices=(8, 16))
    p.add_argument("--seed", type=int, default=20261123)
    p.add_argument("--speed", type=float, default=0.7,
                   help="Translation speed for direction/restart cases (m/s); long_forward stays 1.0")
    p.add_argument("--yaw-rate", type=float, default=0.6,
                   help="Magnitude for left/right rotation cases (rad/s)")
    p.add_argument("--turn-radius", type=float, default=1.0,
                   help="Minimum arc radius (m); arc yaw magnitude=min(yaw-rate, speed/turn-radius)")
    p.add_argument("--profile", choices=("clean", "training", "stress", "all"), default="all",
                   help="clean=nominal/no noise/no pushes; training=checkpoint DR/noise/pushes; stress=shared stairs-inspired DR+all-axis pushes")
    p.add_argument("--sampled", action="store_true", help="Sample policy actions; default uses deterministic means")
    p.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=SCENARIOS)
    p.add_argument("--frames", choices=("none", "key", "all"), default="key",
                   help="Consecutive side-view frames; key=long_forward/restart/direction_switch")
    p.add_argument("--out", required=True, type=Path, help="Output directory (use a fresh directory per invocation)")
    a = p.parse_args()
    if a.seconds < 0.5 or a.wall_seconds < 1:
        p.error("--seconds must be >=0.5 and --wall-seconds must be >=1")
    if not np.isfinite(a.speed) or a.speed <= 0 or not np.isfinite(a.yaw_rate) or a.yaw_rate <= 0:
        p.error("--speed and --yaw-rate must be finite and >0")
    if not np.isfinite(a.turn_radius) or a.turn_radius <= 0:
        p.error("--turn-radius must be finite and >0")
    if a.out.exists() and any(a.out.iterdir()):
        p.error("--out is not empty; choose a fresh directory to preserve previous evidence")
    return a


def main():
    args = parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + args.wall_seconds
    profiles = ["clean", "training", "stress"] if args.profile == "all" else [args.profile]
    report = {"verification_layer": "MuJoCo only", "seed": args.seed,
              "action_mode": "sampled" if args.sampled else "deterministic", "complete": False,
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "metric_notes": ["first episodes only; terminal physical state included before automatic reset",
                               "paths expressed in initial-yaw axes; ideal yaw integrates commanded yaw rate",
                               "failed environments are excluded after their terminal sample; inspect survival alongside errors",
                               "latency requires 0.3 s continuously in tolerance; null means failure or insufficient time",
                               "foot lift requires foot-centre z>0.035m; contact fraction is z<=0.03m height proxy, not force contact",
                               "support drag uses world-foot diff/dt only for both endpoint z<=0.028m and |vz|<0.05m/s; height/low-vz proxy, not proven force-contact slip",
                               "foot events and gait statistics exclude startup; turn tracking and actual yaw rate must be assessed alongside foot lift",
                               "training profile varies with checkpoint; clean/stress conditions are common across checkpoints"],
              "results": []}
    report_path = args.out / "summary.json"
    def save_report():
        report["wall_seconds"] = time.monotonic() - started
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save_report()
    try:
        for index, checkpoint in enumerate(args.checkpoint):
            if time.monotonic() >= deadline:
                raise BudgetExpired("wall deadline before checkpoint load")
            # Reset action RNG for each checkpoint: same draws and initial seed.
            torch.manual_seed(args.seed)
            ac, ck = load_policy(checkpoint)
            for profile in profiles:
                folder = args.out / f"{index}_{Path(checkpoint).parent.name}_{Path(checkpoint).stem}_{profile}"
                folder.mkdir(parents=True, exist_ok=True)
                for scenario in args.scenarios:
                    torch.manual_seed(args.seed)
                    metrics = evaluate(ac, ck, args, profile, scenario, folder, deadline)
                    report["results"].append({"checkpoint": str(Path(checkpoint).resolve()), "profile": profile, **metrics})
                    save_report()
                    print(f"case_done checkpoint={checkpoint} profile={profile} scenario={scenario} "
                          f"survived={metrics['survived_envs']}/{args.num_envs} "
                          f"xy_error={metrics['velocity_xy_error_mps']['mean']} "
                          f"last_path_error={metrics['last_valid_path_error_m']['mean']}", flush=True)
        report["complete"] = True
    except BudgetExpired as exc:
        report["stopped_reason"] = str(exc)
        print(f"DEADLINE: {exc}; completed cases preserved in {report_path}", flush=True)
    except BaseException as exc:
        report["stopped_reason"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save_report()
    print(f"evaluation_complete={report['complete']} cases={len(report['results'])} summary={report_path}", flush=True)
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
