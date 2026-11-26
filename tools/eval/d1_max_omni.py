"""Evaluate a D1 Max omni checkpoint in MuJoCo.

uv run python tools/eval/d1_max_omni.py runs/omni/model_100.pt \
    --stage all --profile promotion --out runs/omni/eval_100

Each scenario starts from a fresh episode. The report scores the first episode
per trial and the trace preserves every sampled step through termination.
"""
import argparse
import csv
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw
import torch

LEGS = ("FBL", "FAR", "RBL", "RAR")
DIAGONALS = (("FBL+RAR", (0, 3)), ("FAR+RBL", (1, 2)))
CLEARANCE_EVENT_M = 0.005
STATE = mujoco.mjtState.mjSTATE_FULLPHYSICS
DIRECTIONS = (
    ("forward", 0), ("forward_left", 45), ("left", 90), ("backward_left", 135),
    ("backward", 180), ("backward_right", 225), ("right", 270), ("forward_right", 315),
)
MANEUVERS = {"left": (1, .10), "right": (1, .10),
             "rotate_left": (2, .24), "rotate_right": (2, .24)}


@dataclass(frozen=True)
class Scenario:
    name: str
    command: tuple[float, float, float]
    start_s: float
    end_s: float
    duration_s: float
    push_deg: int | None = None
    push_s: float | None = None


@dataclass(frozen=True)
class Limits:
    tracking_gain_min: float
    tracking_gain_max: float
    tracking_linear_rmse_mps: float
    tracking_yaw_rmse_radps: float
    tap_angle_gain_min: float
    tap_angle_gain_max: float
    tap_clearance_p95_m: float
    tap_clearance_max_m: float
    tap_release_speed_p95_mps: float
    tap_release_yaw_p95_radps: float
    tap_displacement_max_m: float
    tilt_max_deg: float
    recovery_deadline_s: float
    recovery_linear_mps: float
    recovery_yaw_radps: float
    recovery_fraction: float


LIMITS = {
    "smoke": Limits(0.25, 1.75, 0.26, 0.35, 0.20, 2.00, 0.06, 0.12,
                    0.18, 0.30, 0.18, 40, 2.20, 0.24, 0.35, 0.75),
    "promotion": Limits(0.60, 1.40, 0.15, 0.22, 0.50, 1.50, 0.025, 0.06,
                        0.10, 0.18, 0.09, 25, 1.60, 0.16, 0.25, 0.95),
}


def scenarios(stage, cfg, lateral_speed=None):
    speed = min(cfg.max_speed, 0.45)
    lateral = cfg.max_lateral if lateral_speed is None else lateral_speed
    yaw = min(cfg.max_yaw, 0.55)
    if stage == "tracking":
        cases = []
        for name, deg in DIRECTIONS:
            angle = math.radians(deg)
            command = (speed * round(math.cos(angle), 8), lateral * round(math.sin(angle), 8), 0.)
            cases.append(Scenario(name, command, 0.60, 4.00, 4.80))
        for sign, side in ((1, "left"), (-1, "right")):
            cases.extend((
                Scenario(f"rotate_{side}", (0., 0., sign * yaw), 0.60, 4.00, 4.80),
                Scenario(f"arc_{side}", (0.7 * speed, sign * 0.7 * lateral, sign * yaw),
                         0.60, 4.00, 4.80),
            ))
        return cases
    if stage == "tap":
        return [
            Scenario(f"tap_{side}_{int(duration * 1000)}ms", (0., 0., sign * yaw),
                     0.80, 0.80 + duration, 2.50)
            for side, sign in (("left", 1), ("right", -1))
            for duration in (0.16, 0.32)
        ]
    return [
        Scenario(f"{mode}_{name}", cmd, 0., 4.60, 4.60, deg, 1.60)
        for mode, cmd in (("stand", (0., 0., 0.)), ("moving", (0.65 * speed, 0.60 * lateral, 0.)))
        for name, deg in DIRECTIONS
    ]


def root_fields(model):
    start = 1 + int(model.joint("root").qposadr[0])
    return slice(start, start + 7)


def joint_fields(model):
    return {
        name: 1 + int(model.joint(name).qposadr[0])
        for leg in LEGS for joint in ("ABAD", "HIP", "KNEE")
        for name in (f"{leg}_{joint}_JOINT",)
    }


def rpy(quaternion):
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.stack((roll, pitch, yaw), axis=-1)


def angle_difference(a, b):
    return np.arctan2(np.sin(a - b), np.cos(a - b))


def run_scenario(env, actor, case, stress, seed, push_scale, action_std):
    # Re-seed trials so paired cases see the same randomized starting conditions.
    # push() uses its own state after model sampling and a case-specific azimuth.
    env.rng = np.random.default_rng(seed)
    action_rng = torch.Generator().manual_seed(seed + 1)
    n = len(env.command)
    trials = np.arange(n)
    env.reset(trials)
    if stress:
        env.randomize_models(trials, scale=1.)
    per_step = round(case.duration_s / env.dt)
    sim_t = np.arange(1, per_step + 1) * env.dt
    command = np.zeros((per_step, 3))
    start = round(case.start_s / env.dt)
    end = round(case.end_s / env.dt)
    command[start:end] = case.command
    push_step = None if case.push_s is None else round(case.push_s / env.dt)
    trace = {
        "t": sim_t, "command": command,
        "state": np.full((per_step + 1, n, env.state.shape[1]), np.nan),
        "velocity": np.full((per_step, n, 3), np.nan),
        "clearance": np.full((per_step, n, 4), np.nan),
        "tilt": np.full((per_step, n), np.nan),
        "fall": np.zeros((per_step, n), dtype=bool),
        "done": np.zeros((per_step, n), dtype=bool),
        "visited": np.zeros((per_step, n), dtype=bool),
    }
    trace["state"][0] = env.state
    active = np.ones(n, dtype=bool)
    for i, cmd in enumerate(command):
        if i == push_step and active.any():
            env.push(trials[active], scale=push_scale, azimuth=math.radians(case.push_deg))
        env.command[:] = cmd
        obs, _ = env.observations()
        with torch.inference_mode():
            action = actor(torch.as_tensor(obs, dtype=torch.float32))
            if action_std is not None:
                action = action + torch.randn(action.shape, generator=action_rng) * action_std
            action = action.cpu().numpy()
        _, done, info = env.step(action)
        trace["visited"][i] = active
        trace["velocity"][i, active] = np.column_stack((info["vx"], info["vy"], info["yaw"]))[active]
        trace["clearance"][i, active] = info["clearance"][active]
        trace["tilt"][i, active] = info["tilt"][active]
        trace["fall"][i, active] = info["fall"][active]
        trace["done"][i, active] = done[active]
        for key in ("swing_ref", "step_trigger"):
            if key in info:
                if key not in trace:
                    trace[key] = np.zeros((per_step, *info[key].shape), dtype=bool)
                trace[key][i, active] = info[key][active]
        # Evaluation mode leaves the terminal pose available without resetting.
        trace["state"][i + 1, active] = env.state[active]
        active &= ~done
        if not active.any():
            break
    trace["pose"] = trace["state"][:, :, root_fields(env.model)]
    trace["rpy"] = rpy(trace["pose"][:, :, 3:])
    return trace


class WheelMotion:
    """Recover synchronized contacts and wheel-center positions from control poses."""

    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        self.base = model.body("BASE_LINK").id
        self.floor = model.geom("floor").id
        self.wheels = np.array([model.geom(f"{leg}_FOOT_LINK_collision").id for leg in LEGS])
        self.radius, self.half_width = model.geom_size[self.wheels, :2].T

    def record(self, trace):
        visited = trace["visited"]
        trace["wheel_floor_contact"] = contact = np.zeros((*visited.shape, len(LEGS)), dtype=bool)
        trace["wheel_body_xy"] = xy = np.full((*visited.shape, len(LEGS), 2), np.nan)
        trace["wheel_state_clearance"] = clearance = np.full((*visited.shape, len(LEGS)), np.nan)
        for step, trial in zip(*np.nonzero(visited)):
            state = trace["state"][step + 1, trial]
            if trace["fall"][step, trial] or not np.isfinite(state).all():
                continue
            data = self.data
            mujoco.mj_setState(self.model, data, state, STATE)
            mujoco.mj_fwdPosition(self.model, data)
            pairs = data.contact.geom
            floor_pairs = (pairs == self.floor).any(axis=1)
            contact[step, trial] = np.isin(self.wheels, pairs[floor_pairs])
            relative = data.geom_xpos[self.wheels] - data.xpos[self.base]
            xy[step, trial] = (relative @ data.xmat[self.base].reshape(3, 3))[:, :2]
            axis_z = np.clip(np.abs(data.geom_xmat[self.wheels, 8]), 0., 1.)
            support = self.radius * np.sqrt(1. - axis_z**2) + self.half_width * axis_z
            clearance[step, trial] = np.maximum(0., data.geom_xpos[self.wheels, 2] - support)


def true_runs(mask):
    changes = np.diff(np.r_[False, mask, False].astype(int))
    return zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))


def wheel_motion_metrics(case, trace, dt, *, response_s=0.):
    """Count observed clearance excursions and measure completed airborne strides."""
    start = round(case.start_s / dt)
    end = round((case.end_s + response_s) / dt)
    available = np.isfinite(trace["wheel_body_xy"][start:end]).all(axis=(2, 3))
    air = ~trace["wheel_floor_contact"][start:end] & available[:, :, None]
    clearance = trace["wheel_state_clearance"][start:end]
    clears = clearance >= CLEARANCE_EVENT_M
    xy = trace["wheel_body_xy"][start:end]
    swing_ref = trace.get("swing_ref")
    if swing_ref is not None:
        swing_ref = swing_ref[start:end] & available[:, :, None]
    n_trials = available.shape[1]
    per_wheel = {}
    for k, leg in enumerate(LEGS):
        counts = []
        peaks = []
        strides = []
        for trial in range(n_trials):
            count = 0
            for first, stop in true_runs(air[:, trial, k]):
                peak = finite_stat(clearance[first:stop, trial, k], np.max)
                if peak is None or peak < CLEARANCE_EVENT_M:
                    continue
                count += 1
                peaks.append(peak)
                if first and stop < len(air) and available[[first - 1, stop], trial].all():
                    # A body-relative takeoff-to-landing displacement avoids counting base translation.
                    strides.append(np.linalg.norm(xy[stop, trial, k] - xy[first - 1, trial, k]))
            counts.append(count)
        per_wheel[leg] = {
            "clearance_peak_m": finite_stat(clearance[:, :, k], np.max),
            "clearance_p95_m": p95(clearance[:, :, k]),
            "floor_contact_fraction": finite_stat((~air[:, :, k])[available], np.mean),
            "clearance_bouts_by_trial": counts,
            "clearance_bout_peak_median_m": finite_stat(peaks, np.median),
            "completed_stride_count": len(strides),
            "body_relative_stride_median_m": finite_stat(strides, np.median),
        }
        if swing_ref is not None:
            reference = swing_ref[:, :, k]
            per_wheel[leg].update({
                "reference_swing_samples": int(reference.sum()),
                "airborne_during_reference_fraction": finite_stat(air[:, :, k][reference], np.mean),
                "clearance_during_reference_p95_m": p95(clearance[:, :, k][reference]),
            })
    diagonal_states = np.zeros(available.shape, dtype=int)
    diagonals = {}
    for code, (label, pair) in enumerate(DIAGONALS, 1):
        other = [k for k in range(len(LEGS)) if k not in pair]
        exact = air[:, :, pair].all(axis=2) & ~air[:, :, other].any(axis=2)
        diagonal_states[exact] = code
        diagonals[label] = {
            "swing_samples_by_trial": exact.sum(axis=0).tolist(),
            "swing_bouts_by_trial": [sum(1 for _ in true_runs(exact[:, trial])) for trial in range(n_trials)],
            "both_above_5mm_samples_by_trial": (exact & clears[:, :, pair].all(axis=2)).sum(axis=0).tolist(),
        }
        if swing_ref is not None:
            planned = swing_ref[:, :, pair].all(axis=2) & ~swing_ref[:, :, other].any(axis=2)
            diagonals[label]["reference_swing_samples_by_trial"] = planned.sum(axis=0).tolist()
            diagonals[label]["actual_and_reference_swing_samples_by_trial"] = (exact & planned).sum(axis=0).tolist()
    alternations = []
    for trial in range(n_trials):
        order = diagonal_states[:, trial]
        observed = order[order != 0]
        alternations.append(int(np.count_nonzero(observed[1:] != observed[:-1])))
    velocity = np.where(available[:, :, None], trace["velocity"][start:end], np.nan)
    result = {
        "window_s": [start * dt, end * dt],
        "available_samples_by_trial": available.sum(axis=0).tolist(),
        "measured_planar_speed_mps": finite_stat(np.linalg.norm(velocity[:, :, :2], axis=2), np.mean),
        "measured_vy_mps": finite_stat(velocity[:, :, 1], np.mean),
        "measured_yaw_rate_radps": finite_stat(velocity[:, :, 2], np.mean),
        "peak_clearance_m": finite_stat(clearance, np.max),
        "clearance_event_threshold_m": CLEARANCE_EVENT_M,
        "wheels": per_wheel,
        "diagonal_airborne_pairs": diagonals,
        "diagonal_alternations_by_trial": alternations,
    }
    if "step_trigger" in trace:
        result["step_triggers_by_trial"] = (trace["step_trigger"][start:end] & available).sum(axis=0).tolist()
    return result


def finite_stat(values, statistic):
    values = np.asarray(values)
    samples = values[np.isfinite(values)]
    return None if not samples.size else float(statistic(samples))


def p95(values):
    return finite_stat(values, lambda samples: np.percentile(samples, 95))


def rmse(values):
    return finite_stat(values, lambda samples: np.sqrt(np.mean(samples ** 2)))


def scenario_metrics(stage, case, trace, limits, dt):
    ts = trace["t"]
    velocity = trace["velocity"]
    pose = trace["pose"]
    yaw = trace["rpy"][:, :, 2]
    clearance = trace["clearance"]
    command = np.array(case.command)
    error = velocity - trace["command"][:, None, :]
    linear_error = np.linalg.norm(error[:, :, :2], axis=-1)
    tilt = np.degrees(trace["tilt"])
    finite_visit = np.isfinite(velocity).all(axis=-1) & np.isfinite(clearance).all(axis=-1) & np.isfinite(tilt)
    valid = bool(np.all(finite_visit[trace["visited"]]))
    falls = int(trace["fall"].sum())
    terminations = int(trace["done"].sum())
    metrics = {
        "falls": falls,
        "terminations": terminations,
        "completed_trials": int(np.sum(~trace["done"].any(axis=0))),
        "nonfinite_samples": int(np.sum(trace["visited"] & ~finite_visit)),
        "max_tilt_deg": finite_stat(tilt, np.max),
        "clearance_max_m": finite_stat(clearance, np.max),
    }
    checks = {"finite": valid, "complete_without_fall": terminations == 0}

    def at_most(key, ceiling):
        value = metrics[key]
        checks[key] = value is not None and value <= ceiling

    def in_range(key, lower, upper):
        value = metrics[key]
        checks[key] = value is not None and lower <= value <= upper

    at_most("max_tilt_deg", limits.tilt_max_deg)
    if stage == "tracking":
        settled = (ts >= case.start_s + 1.0) & (ts <= case.end_s + 1e-9)
        metrics["linear_rmse_mps"] = rmse(linear_error[settled])
        metrics["yaw_rmse_radps"] = rmse(error[settled, :, 2])
        at_most("linear_rmse_mps", limits.tracking_linear_rmse_mps)
        at_most("yaw_rmse_radps", limits.tracking_yaw_rmse_radps)
        for j, axis in enumerate(("vx", "vy", "yaw")):
            if abs(command[j]) > 1e-6:
                name = f"{axis}_gain"
                metrics[name] = finite_stat(velocity[settled, :, j] / command[j], np.mean)
                in_range(name, limits.tracking_gain_min, limits.tracking_gain_max)
        begin = round(case.start_s / dt)
        end = round(case.end_s / dt)
        metrics["mean_heading_change_deg"] = finite_stat(
            np.degrees(angle_difference(yaw[end], yaw[begin])), np.mean
        )
    elif stage == "tap":
        start = round(case.start_s / dt)
        release = round(case.end_s / dt)
        response_end = release + round(0.45 / dt)
        released = ts >= case.end_s + 0.65
        footprint = clearance[start:response_end]
        max_by_foot = {leg: finite_stat(footprint[:, :, j], np.max) for j, leg in enumerate(LEGS)}
        per_step_max = np.max(footprint, axis=-1)
        metrics.update({
            "tap_yaw_gain": finite_stat(angle_difference(yaw[response_end], yaw[start]) /
                                        (command[2] * (release - start) * dt), np.median),
            "tap_angle_deg_median": finite_stat(
                np.degrees(angle_difference(yaw[response_end], yaw[start])), np.median
            ),
            "tap_clearance_p95_m": p95(per_step_max),
            "tap_clearance_max_m": finite_stat(per_step_max, np.max),
            "tap_clearance_peak_median_m": finite_stat(np.max(per_step_max, axis=0), np.median),
            "tap_clearance_by_foot_m": max_by_foot,
            "tap_air_fraction_above_15mm": finite_stat(footprint, lambda x: np.mean(x > 0.015)),
            "tap_release_speed_p95_mps": p95(np.linalg.norm(velocity[released, :, :2], axis=-1)),
            "tap_release_yaw_p95_radps": p95(np.abs(velocity[released, :, 2])),
            "tap_displacement_max_m": finite_stat(
                np.linalg.norm(pose[start:, :, :2] - pose[start, :, :2], axis=-1), np.max
            ),
        })
        in_range("tap_clearance_peak_median_m", .008, limits.tap_clearance_max_m)
        in_range("tap_yaw_gain", limits.tap_angle_gain_min, limits.tap_angle_gain_max)
        for key in ("tap_clearance_p95_m", "tap_clearance_max_m", "tap_release_speed_p95_mps",
                    "tap_release_yaw_p95_radps", "tap_displacement_max_m"):
            at_most(key, getattr(limits, key))
    else:
        start = round(case.push_s / dt)
        in_band = ((linear_error[start:] <= limits.recovery_linear_mps) &
                   (np.abs(error[start:, :, 2]) <= limits.recovery_yaw_radps) &
                   (tilt[start:] <= limits.tilt_max_deg))
        hold = round(0.30 / dt)
        recovery = []
        for i, trial_ok in enumerate(in_band.T):
            holds = np.flatnonzero(np.convolve(trial_ok.astype(int), np.ones(hold, dtype=int), mode="valid") == hold)
            latency = (int(holds[0]) + hold) * dt if len(holds) and not trace["done"][:, i].any() else None
            recovery.append(latency)
        metrics.update({
            "push_deg_world": case.push_deg,
            "post_push_peak_linear_error_mps": finite_stat(linear_error[start:], np.max),
            "pre_push_linear_rmse_mps": rmse(linear_error[max(0, start - round(0.4 / dt)):start]),
            "recovery_s_by_trial": recovery,
            "recovered_fraction": float(np.mean([t is not None and t <= limits.recovery_deadline_s
                                                 for t in recovery])),
        })
        checks["recovered_fraction"] = metrics["recovered_fraction"] >= limits.recovery_fraction
    return metrics, checks


def save_trace(path, trace, model, case):
    joints = joint_fields(model)
    fields = [
        "trial", "step", "t_s", "command_vx_mps", "command_vy_mps", "command_yaw_radps",
        "vx_mps", "vy_mps", "yaw_rate_radps", "tilt_deg",
        *(f"{leg}_clearance_m" for leg in LEGS),
        "base_x_m", "base_y_m", "base_z_m", "base_qw", "base_qx", "base_qy", "base_qz",
        "roll_deg", "pitch_deg", "heading_deg",
        *(f"{name}_rad" for name in joints),
        "push_deg_world", "fall", "done",
    ]
    if "wheel_floor_contact" in trace:
        fields.extend(f"{leg}_floor_contact" for leg in LEGS)
        fields.extend(f"{leg}_state_clearance_m" for leg in LEGS)
        fields.extend(f"{leg}_body_{axis}_m" for leg in LEGS for axis in ("x", "y"))
    if "swing_ref" in trace:
        fields.extend(f"{leg}_reference_swing" for leg in LEGS)
    if "step_trigger" in trace:
        fields.append("step_trigger")
    push_step = None if case.push_s is None else round(case.push_s / trace["t"][0])
    rpy_deg = np.degrees(trace["rpy"])
    tilt_deg = np.degrees(trace["tilt"])
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        for i, j in zip(*np.nonzero(trace["visited"])):
            sample = [
                j, i + 1, trace["t"][i], *trace["command"][i], *trace["velocity"][i, j],
                tilt_deg[i, j], *trace["clearance"][i, j],
                *trace["pose"][i + 1, j], *rpy_deg[i + 1, j],
                *(trace["state"][i + 1, j, idx] for idx in joints.values()),
                case.push_deg if i == push_step else None,
                int(trace["fall"][i, j]), int(trace["done"][i, j]),
            ]
            if "wheel_floor_contact" in trace:
                xy = trace["wheel_body_xy"][i, j].ravel()
                contact = trace["wheel_floor_contact"][i, j].astype(int) if np.isfinite(xy).all() else [None] * len(LEGS)
                sample.extend(contact)
                sample.extend(trace["wheel_state_clearance"][i, j])
                sample.extend(xy)
            if "swing_ref" in trace:
                sample.extend(trace["swing_ref"][i, j].astype(int))
            if "step_trigger" in trace:
                sample.append(int(trace["step_trigger"][i, j]))
            writer.writerow([csv_value(value) for value in sample])


def csv_value(value):
    if isinstance(value, (float, np.floating)):
        return f"{value:.7f}" if np.isfinite(value) else ""
    return value


def finite_mean(values, axis):
    finite = np.isfinite(values)
    counts = finite.sum(axis=axis)
    total = np.where(finite, values, 0).sum(axis=axis)
    return np.divide(total, counts, out=np.full(counts.shape, np.nan), where=counts > 0)


def max_across_trials(values):
    count = np.isfinite(values).sum(axis=1)
    top = np.where(np.isfinite(values), values, -np.inf).max(axis=1)
    return np.where(count > 0, top, np.nan)


def save_curves(path, case, trace, passed):
    ts = trace["t"]
    cmd = trace["command"]
    vel = trace["velocity"]
    colors = ("#0571b0", "#149867", "#bc4308")
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 6.3), layout="constrained")
    for j, label in enumerate(("Vx (m/s)", "Vy (m/s)", "Yaw rate (rad/s)")):
        ax = axes[0, j]
        ax.plot(ts, vel[:, :, j], color=colors[j], alpha=0.12, lw=0.65)
        ax.plot(ts, finite_mean(vel[:, :, j], axis=1), color=colors[j], lw=1.3, label="Trial mean")
        ax.step(np.r_[0., ts], np.r_[cmd[0, j], cmd[:, j]], where="pre", color="black", lw=0.9,
                linestyle="--", label="Command")
        ax.set_title(label)
    axes[0, 0].legend(loc="best", fontsize=7)
    for j, leg in enumerate(LEGS):
        axes[1, 0].plot(ts, 1000 * max_across_trials(trace["clearance"][:, :, j]), lw=1, label=leg)
    axes[1, 0].set_title("Wheel clearance (mm), max by trial")
    axes[1, 0].legend(loc="best", fontsize=7, ncol=2)
    axes[1, 1].plot(ts, np.degrees(trace["tilt"]), color="#5a51a1", alpha=0.13, lw=0.65)
    axes[1, 1].plot(ts, np.degrees(max_across_trials(trace["tilt"])), color="#5a51a1", lw=1.3)
    axes[1, 1].set_title("Base tilt (degrees)")
    pose = trace["pose"]
    xy = pose[:, :, :2] - pose[0, :, :2]
    for j in range(xy.shape[1]):
        axes[1, 2].plot(xy[:, j, 0], xy[:, j, 1], lw=0.8, alpha=0.7)
    axes[1, 2].scatter([0], [0], s=12, color="black", zorder=3)
    axes[1, 2].set(xlabel="X (m)", ylabel="Y (m)", title="Base paths, world axes")
    axes[1, 2].set_aspect("equal", adjustable="datalim")
    for ax in (*axes[0], *axes[1, :2]):
        ax.set(xlabel="Time (s)", xlim=(0, ts[-1]))
        if case.start_s > 0:
            ax.axvline(case.start_s, color="0.6", lw=0.6)
            ax.axvline(case.end_s, color="0.6", lw=0.6)
        if case.push_s is not None:
            ax.axvline(case.push_s, color="#a92238", linestyle=":", lw=1)
    for ax in axes.flat:
        ax.grid(alpha=0.20)
    fig.suptitle(f"{case.name.replace('_', ' ').title()}  |  {'PASS' if passed else 'FAIL'}")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_contact_timeline(path, case, trace, dt):
    fig, ax = plt.subplots(figsize=(10, 2.8), layout="constrained")
    available = np.isfinite(trace["wheel_body_xy"][:, 0]).all(axis=(1, 2))
    edges = np.r_[0., trace["t"]]
    for k, leg in enumerate(LEGS):
        airborne = ~trace["wheel_floor_contact"][:, 0, k] & available
        clear = airborne & (trace["wheel_state_clearance"][:, 0, k] >= CLEARANCE_EVENT_M)
        bands = [(airborne, "#8fbcd1", -.26, .52), (clear, "#126196", -.26, .52)]
        if "swing_ref" in trace:
            bands.append((trace["swing_ref"][:, 0, k] & available, "#bc4308", .31, .10))
        for samples, color, offset, height in bands:
            intervals = [(edges[first], (stop - first) * dt) for first, stop in true_runs(samples)]
            ax.broken_barh(intervals, (k + offset, height), facecolor=color)
    ax.axvspan(case.start_s, case.end_s, color="0.2", alpha=.06, zorder=-1)
    ax.set(yticks=range(len(LEGS)), yticklabels=LEGS, xlabel="Time (s)", xlim=(0, edges[-1]),
           title=f"{case.name.replace('_', ' ').title()} · trial 0 wheel–floor contact")
    labels = [("#8fbcd1", "No floor contact"), ("#126196", "Clearance ≥ 5 mm")]
    if "swing_ref" in trace:
        labels.append(("#bc4308", "Reference swing"))
    for color, label in labels:
        ax.plot([], [], color=color, lw=4, label=label)
    ax.legend(loc="upper right", ncol=len(labels), fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.20)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_overview(path, stage, records, limits):
    cases = list(records)
    y = np.arange(len(cases))
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(12, max(3.3, 0.35 * len(cases) + 1.1)),
                             layout="constrained")
    colors = ["#246a4f" if records[name]["pass"] else "#b23e3e" for name in cases]

    def values(key, scale=1):
        return [
            np.nan if (value := records[name]["metrics"].get(key)) is None else scale * value
            for name in cases
        ]

    def bars(ax, key, label, limit=None, scale=1):
        ax.barh(y, values(key, scale), color=colors, height=0.60)
        if limit is not None:
            ax.axvline(scale * limit, color="0.25", linestyle="--", lw=0.8)
        ax.set_title(label, fontsize=10)

    def gain_band(ax, lower, upper):
        ax.axvspan(lower, upper, color="#e2efe7")
        ax.axvline(1, color="0.25", linestyle="--", lw=0.8)

    if stage == "tracking":
        ax = axes[0]
        gain_band(ax, limits.tracking_gain_min, limits.tracking_gain_max)
        for axis, color, marker in (("vx", "#126196", "o"), ("vy", "#149867", "s"), ("yaw", "#bc4308", "^")):
            ax.scatter(values(f"{axis}_gain"), y, label=axis, color=color, marker=marker, s=21, zorder=3)
        ax.set_title("Command gain", fontsize=10)
        ax.legend(loc="best", fontsize=7)
        bars(axes[1], "linear_rmse_mps", "Planar RMSE (m/s)", limits.tracking_linear_rmse_mps)
        bars(axes[2], "yaw_rmse_radps", "Yaw rate RMSE (rad/s)", limits.tracking_yaw_rmse_radps)
    elif stage == "tap":
        ax = axes[0]
        gain_band(ax, limits.tap_angle_gain_min, limits.tap_angle_gain_max)
        ax.scatter(values("tap_yaw_gain"), y, c=colors, s=24, zorder=3)
        ax.set_title("Yaw angle gain", fontsize=10)
        bars(axes[1], "tap_clearance_p95_m", "Wheel clearance p95 (mm)", limits.tap_clearance_p95_m, 1000)
        bars(axes[2], "tap_clearance_max_m", "Wheel clearance peak (mm)", limits.tap_clearance_max_m, 1000)
    else:
        bars(axes[0], "recovered_fraction", "Recovered by deadline (%)", limits.recovery_fraction, 100)
        axes[0].set_xlim(0, 105)
        bars(axes[1], "post_push_peak_linear_error_mps", "Peak planar error (m/s)")
        bars(axes[2], "max_tilt_deg", "Peak tilt (degrees)", limits.tilt_max_deg)

    axes[0].set_yticks(y, [name.replace("_", " ") for name in cases], fontsize=8)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(axis="x", alpha=0.18)
    fig.suptitle(f"{stage.title()} evaluation")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def closest_valid_window(states, trial, center, n_frames):
    last = np.flatnonzero(np.isfinite(states[:, trial]).all(axis=1))[-1]
    width = min(n_frames, last + 1)
    start = max(0, min(center - width // 2, last - width + 1))
    return range(start, start + width)


class PoseRenderer:
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        self.renderer = mujoco.Renderer(model, height=240, width=400)
        self.camera = mujoco.MjvCamera()
        self.camera.distance = 1.70
        self.root = root_fields(model)

    def close(self):
        self.renderer.close()

    def frame(self, center, azimuth, elevation, distance=1.70):
        self.camera.lookat[:] = (center[0], center[1], max(0.22, center[2] * 0.65))
        self.camera.azimuth = azimuth
        self.camera.elevation = elevation
        self.camera.distance = distance
        self.renderer.update_scene(self.data, camera=self.camera)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        return Image.fromarray(self.renderer.render())

    def save(self, folder, trace, trial, center, n_frames, dt, label):
        folder.mkdir(parents=True, exist_ok=True)
        frames = []
        state = trace["state"]
        for i in closest_valid_window(state, trial, center, n_frames):
            pose = state[i, trial]
            mujoco.mj_setState(self.model, self.data, pose, STATE)
            mujoco.mj_forward(self.model, self.data)
            root = pose[self.root]
            views = (self.frame(root, 90, -8), self.frame(root, 150, -22), self.frame(root, 90, -90, 1.35))
            width, height = views[0].size
            frame = Image.new("RGB", (len(views) * width, height + 24), "#ffffff")
            for index, view in enumerate(views):
                frame.paste(view, (index * width, 24))
            draw = ImageDraw.Draw(frame)
            draw.text((7, 6), f"{label} | trial {trial} | control step {i} | {i * dt:.3f} s",
                      fill="#101010")
            draw.text((width + 7, 6), "Three-quarter view", fill="#101010")
            overhead = "Overhead"
            if i and "wheel_floor_contact" in trace and np.isfinite(trace["wheel_body_xy"][i - 1, trial]).all():
                on_floor = [leg for leg, contact in zip(LEGS, trace["wheel_floor_contact"][i - 1, trial]) if contact]
                overhead += " | Floor contacts " + (", ".join(on_floor) if on_floor else "none")
            draw.text((2 * width + 7, 6), overhead, fill="#101010")
            frame.save(folder / f"step_{i:04d}.png")
            frames.append(frame)
        rows = math.ceil(len(frames) / 2)
        width, height = frames[0].size
        sheet = Image.new("RGB", (2 * width, rows * height), "#ffffff")
        for i, frame in enumerate(frames):
            sheet.paste(frame, (i % 2 * width, i // 2 * height))
        sheet.save(folder / "montage.png")


def trace_peak(values, start=0):
    block = values[start:]
    finite = np.isfinite(block)
    if not finite.any():
        return start, 0
    i, trial = np.unravel_index(np.where(finite, block, -np.inf).argmax(), block.shape[:2])
    return i + start + 1, trial


def maneuver_pose_center(case, trace, dt):
    start = round(case.start_s / dt)
    end = round(case.end_s / dt)
    contact = trace["wheel_floor_contact"][:end, :1]
    diagonal = ~contact[:, :, [1, 2]].any(axis=2) & contact[:, :, [0, 3]].all(axis=2)
    left_rear = trace["wheel_state_clearance"][:end, :1, 2]
    diagonal_clearance = np.where(diagonal, left_rear, np.nan)
    if np.isfinite(diagonal_clearance[start:]).any():
        return trace_peak(diagonal_clearance, start)[0]
    return trace_peak(left_rear, start)[0]


def save_pose_evidence(renderer, stage, results, folder, dt, n_frames):
    if stage == "tracking":
        selected = {"forward", "backward", "forward_left", "backward_right", "arc_left", *MANEUVERS}
        for case, trace, _ in results:
            if case.name in selected:
                renderer.save(folder / case.name / "settled_pose", trace, 0,
                              round(2.4 / dt), n_frames, dt, "Settled command")
            if case.name in MANEUVERS:
                renderer.save(folder / case.name / "maneuver_pose", trace, 0,
                              maneuver_pose_center(case, trace, dt), n_frames, dt, "Maneuver clearance")
    elif stage == "tap":
        for case, trace, _ in results:
            start = round(case.start_s / dt)
            release = round(case.end_s / dt)
            end = release + round(0.45 / dt)
            peak, trial = trace_peak(trace["clearance"].max(axis=2)[:end], start)
            renderer.save(folder / case.name / "tap_onset", trace, trial, start, n_frames, dt, "Tap onset")
            renderer.save(folder / case.name / "tap_release", trace, trial, release, n_frames, dt, "Release")
            if abs(peak - start) > n_frames // 2 and abs(peak - release) > n_frames // 2:
                renderer.save(folder / case.name / "clearance_peak", trace, trial, peak,
                              n_frames, dt, "Peak clearance")
    else:
        for mode in ("stand", "moving"):
            cases = [item for item in results if item[0].name.startswith(f"{mode}_")]
            case, trace, _ = max(
                cases, key=lambda item: item[2]["metrics"]["max_tilt_deg"] or 0.
            )
            start = round(case.push_s / dt)
            peak, trial = trace_peak(trace["tilt"], start)
            base = folder / case.name
            renderer.save(base / "impulse", trace, trial, start + n_frames // 4, n_frames, dt, "Impulse")
            if abs(peak - start) > n_frames // 2:
                renderer.save(base / "recovery_peak", trace, trial, peak, n_frames, dt, "Post-push tilt peak")


def write_json(path, data):
    pending = path.with_name(f".{path.name}.tmp")
    pending.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def save_training_curves(checkpoint, output):
    source = checkpoint.parent / "metrics.csv"
    if not source.is_file():
        return
    with source.open(newline="") as log:
        rows = list(csv.DictReader(log))
    if not rows:
        return
    iterations = [int(row["iteration"]) for row in rows]
    panels = (("reward", "Reward"), ("vx_mae", "Vx error (m/s)"), ("vy_mae", "Vy error (m/s)"),
              ("yaw_mae", "Yaw error (rad/s)"), ("lift_mm", "Mean peak clearance (mm)"), ("falls", "Falls"))
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    for ax, (column, label) in zip(axes.flat, panels):
        ax.plot(iterations, [float(row[column]) for row in rows], linewidth=1)
        ax.set(xlabel="PPO iteration", ylabel=label)
        ax.grid(alpha=.25)
    fig.savefig(output, dpi=130)
    plt.close(fig)


def save_speed_comparison(env, actor, motion, results, folder, args, renderer, action_std):
    comparison = {}
    for high, _, record in results:
        if high.name not in MANEUVERS:
            continue
        axis, nominal_low = MANEUVERS[high.name]
        command = list(high.command)
        command[axis] = math.copysign(min(nominal_low, .5 * abs(command[axis])), command[axis])
        low = replace(high, name=f"{high.name}_slow", command=tuple(command))
        case_dir = folder / low.name
        case_dir.mkdir(exist_ok=True)
        trace = run_scenario(env, actor, low, False, args.seed, args.push_scale, action_std)
        motion.record(trace)
        metrics = wheel_motion_metrics(low, trace, env.dt)
        save_trace(case_dir / "trace.csv", trace, env.model, low)
        save_contact_timeline(case_dir / "wheel_contacts.png", low, trace, env.dt)
        if renderer is not None:
            renderer.save(case_dir / "maneuver_pose", trace, 0, maneuver_pose_center(low, trace, env.dt),
                          args.frames, env.dt, "Slow maneuver clearance")
        deltas = {}
        for leg in LEGS:
            fast_wheel = record["wheel_motion"]["wheels"][leg]
            slow_wheel = metrics["wheels"][leg]
            deltas[leg] = {}
            for key in ("clearance_peak_m", "clearance_bout_peak_median_m", "body_relative_stride_median_m"):
                fast, slow = fast_wheel[key], slow_wheel[key]
                deltas[leg][key] = None if fast is None or slow is None else fast - slow
        comparison[high.name] = {
            "slow": {"scenario": asdict(low), "wheel_motion": metrics},
            "fast_scenario": high.name,
            "fast_minus_slow_by_wheel_m": deltas,
        }
    return comparison


def evaluate(args, actor, cfg, env_type, action_std):
    stages = ("tracking", "tap", "stress") if args.stage == "all" else (args.stage,)
    limits = LIMITS[args.profile]
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "profile": args.profile,
        "seed": args.seed,
        "trials_per_scenario": args.num_envs,
        "lateral_speed_override_mps": args.lateral_speed,
        "sampled_policy": action_std is not None,
        "thresholds": asdict(limits),
        "config": asdict(cfg),
        "stages": {},
        "all_selected_passed": False,
    }
    eval_cfg = replace(cfg, episode_s=max(6., cfg.episode_s))
    env = env_type(args.num_envs, eval_cfg, seed=args.seed, nthread=args.nthread, training=False)
    wheel_motion = WheelMotion(env.model) if any(stage != "stress" for stage in stages) else None
    visual_env = None
    renderer = None
    try:
        if not args.no_render:
            visual_env = env_type(1, eval_cfg, seed=args.seed, training=False, visual=True)
            renderer = PoseRenderer(visual_env.model)
        for stage in stages:
            folder = args.out / stage
            folder.mkdir(exist_ok=True)
            stage_report = {"pass": False, "complete": False, "domain_randomized": stage == "stress",
                            "scenarios": {}}
            if stage in ("tracking", "tap"):
                stage_report["wheel_motion_measurement"] = {
                    "method": "MuJoCo wheel–floor geom contact pairs at control-frame poses",
                    "left_rear_wheel": "RBL",
                    "stride": "Planar distance from last contact before lift to first contact after landing, in body coordinates",
                    "clearance_bout": "Consecutive frames without floor contact with a clearance peak of at least 5 mm",
                }
            if stage == "stress":
                stage_report.update({
                    "push_scale": args.push_scale,
                    "push_xy_mps": cfg.push_xy * args.push_scale,
                    "push_z_max_mps": cfg.push_z * args.push_scale,
                    "push_angular_max_radps": cfg.push_angular * args.push_scale,
                })
            results = []
            for case in scenarios(stage, cfg, args.lateral_speed):
                case_dir = folder / case.name
                case_dir.mkdir(exist_ok=True)
                trace = run_scenario(env, actor, case, stage == "stress", args.seed, args.push_scale, action_std)
                metrics, gates = scenario_metrics(stage, case, trace, limits, env.dt)
                record = {"pass": all(gates.values()), "scenario": asdict(case),
                          "metrics": metrics, "checks": gates}
                if stage == "tap" or (stage == "tracking" and case.name in MANEUVERS):
                    wheel_motion.record(trace)
                    record["wheel_motion"] = wheel_motion_metrics(
                        case, trace, env.dt, response_s=0.45 if stage == "tap" else 0.0,
                    )
                    save_contact_timeline(case_dir / "wheel_contacts.png", case, trace, env.dt)
                stage_report["scenarios"][case.name] = record
                results.append((case, trace, record))
                save_trace(case_dir / "trace.csv", trace, env.model, case)
                save_curves(case_dir / "curves.png", case, trace, record["pass"])
                write_json(folder / "metrics.json", stage_report)
                failures = ", ".join(key for key, ok in gates.items() if not ok)
                print(f"{stage:8} {case.name:24} {'PASS' if record['pass'] else 'FAIL'} {failures}", flush=True)
            if stage == "tracking":
                stage_report["speed_comparison"] = save_speed_comparison(
                    env, actor, wheel_motion, results, folder, args, renderer, action_std,
                )
            save_overview(folder / "overview.png", stage, stage_report["scenarios"], limits)
            if renderer is not None:
                save_pose_evidence(renderer, stage, results, folder, env.dt, args.frames)
            stage_report["pass"] = all(item[2]["pass"] for item in results)
            stage_report["complete"] = True
            write_json(folder / "metrics.json", stage_report)
            summary["stages"][stage] = stage_report
            summary["all_selected_passed"] = len(summary["stages"]) == len(stages) and all(
                report["pass"] for report in summary["stages"].values()
            )
            write_json(args.out / "summary.json", summary)
            print(f"{stage}: {'PASS' if stage_report['pass'] else 'FAIL'}", flush=True)
    finally:
        try:
            if renderer is not None:
                renderer.close()
        finally:
            if visual_env is not None:
                visual_env.close()
            env.close()
    return summary["all_selected_passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--stage", choices=("tracking", "tap", "stress", "all"), default="all")
    parser.add_argument("--profile", choices=tuple(LIMITS), default="promotion")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=83371)
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Trials per scenario (default: 4 for smoke, 12 for promotion)")
    parser.add_argument("--nthread", type=int, default=4)
    parser.add_argument("--push-scale", type=float, default=1.)
    parser.add_argument("--lateral-speed", type=float, help="Lateral target in m/s; defaults to the checkpoint cap")
    parser.add_argument("--frames", type=int, default=10, help="Consecutive control frames per pose window")
    parser.add_argument("--no-render", action="store_true", help="Save CSV, metrics and curves without 3D pose PNGs")
    parser.add_argument("--sampled", action="store_true", help="Evaluate with checkpoint exploration standard deviations")
    parser.add_argument("--require-pass", action="store_true", help="Return a nonzero exit status if any gate fails")
    args = parser.parse_args()
    if args.num_envs is None:
        args.num_envs = 4 if args.profile == "smoke" else 12
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.num_envs < 1 or args.nthread < 0 or args.seed < 0 or args.frames < 2:
        parser.error("num-envs must be positive, frames must be at least 2, seed and nthread cannot be negative")
    if not math.isfinite(args.push_scale) or args.push_scale <= 0:
        parser.error("push-scale must be finite and positive")
    if args.out is None:
        args.out = args.checkpoint.parent / f"eval_{args.checkpoint.stem}_{args.stage}_{args.profile}"
    args.out.mkdir(parents=True, exist_ok=True)
    from rl01.d1_max_omni import OmniEnv
    from rl01.train_d1_max_omni import load_policy, read_checkpoint

    torch.set_num_threads(1)
    actor, cfg = load_policy(str(args.checkpoint))
    if min(cfg.max_speed, cfg.max_lateral, cfg.max_yaw) <= 0:
        parser.error("checkpoint must provide positive speed, lateral and yaw limits")
    if args.lateral_speed is not None and not 0 < args.lateral_speed <= cfg.max_lateral:
        parser.error("lateral-speed must be positive and at most the checkpoint's lateral limit")
    if args.stage in ("all", "stress") and (not math.isfinite(cfg.push_xy) or cfg.push_xy <= 0):
        parser.error("stress requires a positive horizontal impulse in the checkpoint configuration")
    action_std = read_checkpoint(args.checkpoint)["model"]["std"].clamp(min=.001) if args.sampled else None
    save_training_curves(args.checkpoint, args.out / "training_curves.png")
    passed = evaluate(args, actor.cpu().eval(), cfg, OmniEnv, action_std)
    print(args.out / "summary.json", flush=True)
    return int(args.require_pass and not passed)


if __name__ == "__main__":
    raise SystemExit(main())
