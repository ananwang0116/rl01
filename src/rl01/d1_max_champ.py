"""State-triggered diagonal foot placement for the D1 Max wheel-leg model."""
import numpy as np

DIAGONALS = np.array([[True, False, False, True], [False, True, True, False]])
SUPPORT_S = .04
SWING_S = .16
TURN_TRIGGER_RAD = np.radians(8.0)
MOVING_TURN_TRIGGER_RAD = np.radians(16.0)


def angle_delta(target, reference):
    return (target - reference + np.pi) % (2 * np.pi) - np.pi


class WheelLegIK:
    """Three-axis wheel-center IK using the MJCF hip and link offsets."""
    def __init__(self, model, data, legs, home, wheel_geoms):
        bodies = lambda suffix: np.array([model.body(f"{leg}_{suffix}_LINK").pos for leg in legs])
        hips, thighs, knees, wheels = (bodies(suffix) for suffix in ("ABAD", "HIP", "KNEE", "FOOT"))
        self.home_center = data.geom_xpos[wheel_geoms] - data.qpos[:3] - hips
        self.thigh_x = thighs[:, 0]
        self.side_y = (thighs + knees + wheels + model.geom_pos[wheel_geoms])[:, 1]
        self.thigh_len = -knees[:, 2]
        self.shin_len = -wheels[:, 2]
        self.knee_sign = np.sign(home.reshape(4, 3)[:, 2])
        self.home_correction = home.reshape(4, 3) - self.solve(self.home_center[None, :])[0]

    def solve(self, center):
        x, y, z = np.moveaxis(center, -1, 0)
        down = np.sqrt(np.maximum(y*y + z*z - self.side_y**2, 1e-8))
        abduct = angle_delta(np.arctan2(z, y), np.arctan2(-down, self.side_y))
        x = self.thigh_x - x
        a, b = self.thigh_len, self.shin_len
        cos_knee = np.clip((x*x + down*down - a*a - b*b) / (2 * a * b), -.9999, .9999)
        knee = self.knee_sign * np.arccos(cos_knee)
        thigh = np.arctan2(x, down) - np.arctan2(b * np.sin(knee), a + b * cos_knee)
        return np.stack((abduct, thigh, knee), axis=-1)

    def targets(self, offsets):
        return (self.solve(self.home_center + offsets) + self.home_correction).reshape(-1, 12)


class DiagonalGait:
    def __init__(self, n, dt, foot_xy, refine=False):
        self.dt = dt
        self.foot_xy = foot_xy
        self.refine = refine
        self.anchor = np.zeros((n, 2))
        self.anchor_yaw = np.zeros(n)
        self.heading_goal = np.zeros(n)
        self.remaining = np.zeros(n)
        self.pair = np.zeros(n, dtype=np.int8)
        self.lateral_mode = np.zeros(n, dtype=bool)
        self.followup = np.zeros(n, dtype=bool)
        self.started = np.zeros(n, dtype=bool)
        self.finished = np.zeros(n, dtype=bool)
        self.trigger = np.zeros(n, dtype=bool)
        self.swing = np.zeros((n, 4), dtype=bool)
        self.motion = np.zeros((n, 2))
        self.offset = np.zeros((n, 4, 2))
        self.origin = np.zeros_like(self.offset)
        self.goal = np.zeros_like(self.offset)
        self.lift = np.zeros(n)
        self.turn_age = np.zeros(n)
        self.wheel_turn = np.zeros(n)

    def reset(self, ids, xy, yaw):
        self.anchor[ids] = xy
        self.anchor_yaw[ids] = self.heading_goal[ids] = yaw
        self.remaining[ids] = 0
        self.pair[ids] = 0
        self.lateral_mode[ids] = False
        self.followup[ids] = False
        self.started[ids] = False
        self.finished[ids] = False
        self.trigger[ids] = False
        self.swing[ids] = False
        self.motion[ids] = 0
        self.offset[ids] = self.origin[ids] = self.goal[ids] = 0
        self.lift[ids] = 0
        self.turn_age[ids] = self.wheel_turn[ids] = 0

    def plan(self, command, xy, yaw, roll_proxy, shoulder_roll, limits):
        vx, vy, wz = command.T
        dt = self.dt
        finished = self.finished
        self.pair[finished] ^= 1
        self.anchor[finished] = xy[finished]
        self.anchor_yaw[finished] = yaw[finished]
        # When the second follow-up turn step finishes and no turn input remains, settle cleanly.
        turn_settled = finished & (~self.lateral_mode) & ~self.followup & (np.abs(wz) <= .045)
        self.started[turn_settled] = False

        moving_lin = np.abs(vx) > .20
        lateral = np.abs(vy) > .040
        self.heading_goal = yaw + np.clip(angle_delta(self.heading_goal, yaw) + wz * dt, -.24, .24)
        heading_error = angle_delta(self.heading_goal, yaw)
        turning = (~lateral) & (
            (np.abs(wz) > .045) | (np.abs(heading_error) > .018) | self.followup | (self.remaining > 0)
        )
        demand = np.column_stack((vy, np.clip(wz + 3.5 * heading_error, -.75, .75)))
        demand[~turning, 1] = 0
        self.motion += min(1., dt / .09) * (demand - self.motion)
        twist = self.motion[:, 1]

        heading_delta = angle_delta(yaw, self.anchor_yaw)
        skew = np.where(
            (~lateral) & (~moving_lin),
            np.clip(angle_delta(self.heading_goal, self.anchor_yaw), -.22, .22),
            0.,
        )
        turn_thresh = np.where(moving_lin, MOVING_TURN_TRIGGER_RAD, TURN_TRIGGER_RAD)
        fast_inplace_turn = (~moving_lin) & (np.abs(wz) >= .35) & self.started
        turn_strain = (
            (np.abs(heading_delta) >= turn_thresh)
            | ((~moving_lin) & (np.abs(skew) >= np.radians(10.0)))
        ) & ((np.abs(wz) > .045) | (np.abs(heading_error) > .02))

        turn_trigger = (~lateral) & (turn_strain | self.followup | fast_inplace_turn)
        self.trigger[:] = (self.remaining == 0) & (lateral | turn_trigger)
        mode_switch = self.trigger & (~self.started | (lateral != self.lateral_mode))
        self.lateral_mode[self.trigger] = lateral[self.trigger]

        # Left lateral (vy > 0) -> pair 1 (FAR + RBL) first, then pair 0 (FBL + RAR).
        # Right lateral (vy < 0) -> pair 0 (FBL + RAR) first, then pair 1 (FAR + RBL).
        # Left turn (wz > 0) -> pair 0 (FBL + RAR) first, then mandatory follow-up pair 1 (FAR + RBL).
        # Right turn (wz < 0) -> pair 1 (FAR + RBL) first, then mandatory follow-up pair 0 (FBL + RAR).
        lat_first = (vy[mode_switch] > 0).astype(np.int8)
        turn_first = (self.motion[mode_switch, 1] < 0).astype(np.int8)
        self.pair[mode_switch] = np.where(lateral[mode_switch], lat_first, turn_first)

        # Arm mandatory second diagonal step when the first step of an in-place turn starts.
        is_followup_dispatch = self.trigger & (~lateral) & self.followup
        is_turn_first = self.trigger & (~lateral) & (~self.followup) & (~moving_lin) & (~fast_inplace_turn)
        self.followup[is_followup_dispatch] = False
        self.followup[is_turn_first] = True
        self.started[self.trigger] = True

        self.remaining[self.trigger] = SUPPORT_S + SWING_S
        self.origin[self.trigger] = self.offset[self.trigger]

        foot_velocity = np.empty_like(self.offset)
        foot_velocity[:, :, 0] = -twist[:, None] * self.foot_xy[:, 1]
        foot_velocity[:, :, 1] = self.motion[:, :1] + twist[:, None] * self.foot_xy[:, 0]

        v_sweep = 3.4 * self.motion[:, 0]
        stride_y = np.clip(v_sweep * SWING_S, -.14, .14)
        lat_goal = np.zeros_like(self.offset)
        lat_goal[:, :, 1] = .50 * stride_y[:, None]
        turn_goal = np.where(
            fast_inplace_turn[:, None, None],
            np.clip(.10 * foot_velocity, -.05, .05),
            0.,
        )
        planned_goal = np.where(self.lateral_mode[:, None, None], lat_goal, turn_goal)
        self.goal[self.trigger] = planned_goal[self.trigger]

        lat_speed = np.clip(np.abs(self.motion[:, 0]) / max(limits[0], 1e-4), 0., 1.)
        yaw_speed = np.clip(np.abs(wz) / max(limits[1], 1e-4), 0., 1.)
        fly = np.where(
            self.lateral_mode,
            .038 + .018 * lat_speed,
            .022 + .018 * yaw_speed,
        )
        self.lift[self.trigger] = fly[self.trigger]

        progress = np.clip((SUPPORT_S + SWING_S - self.remaining - SUPPORT_S) / SWING_S, 0., 1.)
        active = self.remaining > 0
        self.swing[:] = active[:, None] & (progress[:, None] > 0) & DIAGONALS[self.pair]

        # Lateral stance legs sweep ONLY while the other diagonal is swinging (never during SUPPORT_S).
        swinging_now = self.swing.any(axis=1)
        stance_lat = (swinging_now & self.lateral_mode)[:, None, None] & ~self.swing[:, :, None]
        self.offset[:, :, 1] -= dt * v_sweep[:, None] * stance_lat[:, :, 0]
        np.clip(self.offset, -.115, .115, out=self.offset)

        alpha = (progress**2 * (3 - 2 * progress))[:, None, None]
        swing_xy = self.origin + alpha * (self.goal - self.origin)
        self.offset[:] = np.where(self.swing[:, :, None], swing_xy, self.offset)

        # Phase 1 of in-place turn: keep all 4 feet planted while torso skews smoothly (01 -> 02 -> 03).
        phase1_skew = (~active) & turning & (~lateral) & (~moving_lin)
        c, s = np.cos(-skew)[:, None], np.sin(-skew)[:, None]
        skew_xy = np.stack((
            self.foot_xy[:, 0] * (c - 1.) - self.foot_xy[:, 1] * s,
            self.foot_xy[:, 0] * s + self.foot_xy[:, 1] * (c - 1.),
        ), axis=-1)
        self.offset[phase1_skew] += min(1., dt / .06) * (skew_xy[phase1_skew] - self.offset[phase1_skew])

        idle = (~active) & (~phase1_skew)
        self.offset[idle] *= .85
        self.anchor[idle & ~turning] = xy[idle & ~turning]
        idle_settled = (~active) & (~turning) & (~lateral)
        self.started[idle_settled] = False
        self.anchor_yaw[idle_settled] = yaw[idle_settled]

        z = (16 * self.lift * progress**2 * (1 - progress)**2)[:, None] * self.swing
        # Gentle stance extension unloads the swinging diagonal without rolling the chassis.
        z -= .008 * self.swing[:, [1, 0, 3, 2]] * np.sin(np.pi * progress)[:, None]
        # While moving forward and turning, keep a slight inward lean posture.
        curve_lean = np.where(moving_lin, np.clip(vx * wz / 1.2, -1., 1.), 0.)
        z[:, [0, 2]] += .010 * curve_lean[:, None]
        z[:, [1, 3]] -= .010 * curve_lean[:, None]
        offsets = np.concatenate((self.offset, z[:, :, None]), axis=2)

        wheel_turn = np.clip(wz + (3.5 if self.refine else 4.2) * heading_error, -.7, .7)
        wheel_turn[lateral | ((np.abs(wz) < .04) & (np.abs(heading_error) < .016))] = 0
        wheel_turn *= np.where(self.swing.any(axis=1), .75, 1.)
        if self.refine:
            self.wheel_turn += np.clip(wheel_turn - self.wheel_turn, -.10, .10)
            wheel_turn = self.wheel_turn
        self.finished[:] = active & (self.remaining <= dt + 1e-8)
        self.remaining[:] = np.where(self.finished, 0., self.remaining - dt * active)
        return offsets, wheel_turn, heading_error
