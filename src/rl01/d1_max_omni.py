"""D1 Max 平地全向控制。轮毂负责纵向滚动，低摆腿轨迹提供横移和点按转向的支撑切换。"""
import copy
from dataclasses import dataclass

import mujoco
from mujoco import rollout
import numpy as np

from rl01.d1_max_env import DECIMATION, JOINT_NAMES, LEGS, LEG_MIRROR, N_ACT, STATE, mirror_action
from rl01.d1_max_champ import DiagonalGait, SUPPORT_S, SWING_S, WheelLegIK
from rl01.env import quat_to_R
from rl01.sim import D1_MAX_SCENE

N_OBS, N_CRITIC_OBS = 60, 64


@dataclass(frozen=True)
class OmniConfig:
    max_speed: float = 0.65
    max_reverse: float | None = None
    wheel_speed_limit: float = 15.0
    max_lateral: float = 0.23
    max_yaw: float = 0.55
    episode_s: float = 8.0
    robustness: float = 0.25
    push_xy: float = 0.45
    push_z: float = 0.10
    push_angular: float = 0.27
    lift_m: float = 0.018
    lateral_lift_m: float = 0.038
    lateral_gait_hz: float = 1.8
    tap_step_s: float = 0.38
    corrective_gait: bool = False
    champ_gait: bool = False
    champ_ik: bool = False
    champ_refine: bool = False
    vy_turn_only: bool = False


def mirror_obs(obs):
    mapped = obs.clone()
    mapped[..., :9] *= obs.new_tensor((-1, 1, -1, 1, -1, 1, 1, -1, -1))
    mapped[..., 9:21] = obs[..., 9:21][..., LEG_MIRROR] * obs.new_tensor((-1, 1, 1) * 4)
    mapped[..., 21:37] = mirror_action(obs[..., 21:37])
    mapped[..., 37:53] = mirror_action(obs[..., 37:53])
    mapped[..., 53:55] = -obs[..., 53:55]  # 左右对换，交替支撑相位平移半周期
    mapped[..., 55:59] = obs[..., [56, 55, 58, 57]]
    mapped[..., 59] = -obs[..., 59]
    return mapped


def build_model(visual, *, world_boxes=()):
    spec = mujoco.MjSpec.from_file(str(D1_MAX_SCENE))
    if not visual:
        for geom in list(spec.geoms):
            if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
                spec.delete(geom)
        for mesh in list(spec.meshes):
            spec.delete(mesh)
    for i, box in enumerate(world_boxes):
        spec.worldbody.add_geom(name=f"stair{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                pos=box["pos"], size=box["size"], rgba=box["rgba"])
    for leg in LEGS:
        geom = f"{leg}_FOOT_LINK_collision"
        spec.add_sensor(name=f"{leg}_center", type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
                        objtype=mujoco.mjtObj.mjOBJ_GEOM, objname=geom)
        spec.add_sensor(name=f"{leg}_orientation", type=mujoco.mjtSensor.mjSENS_FRAMEQUAT,
                        objtype=mujoco.mjtObj.mjOBJ_GEOM, objname=geom)
    return spec.compile()


class OmniEnv:
    def __init__(self, num_envs, cfg=None, *, seed=0, nthread=0, visual=False, training=True, world_boxes=()):
        self.cfg = cfg or OmniConfig()
        self.training = training
        self.n = num_envs
        self.rng = np.random.default_rng(seed)
        self.model = model = build_model(visual, world_boxes=world_boxes)
        self.dt = model.opt.timestep * DECIMATION
        self.max_steps = max(1, round(self.cfg.episode_s / self.dt))
        self.base = model.body("BASE_LINK").id
        self.act_ids = np.array([model.actuator(name).id for name in JOINT_NAMES])
        joints = np.array([model.joint(name).id for name in JOINT_NAMES])
        if not np.array_equal(model.actuator_trnid[self.act_ids, 0], joints):
            raise ValueError("D1 Max actuator transmission mismatch")
        self.wheel_ids = self.act_ids[3::4]
        self.leg_ids = np.delete(self.act_ids, [3, 7, 11, 15])
        self.leg_actions = np.delete(np.arange(N_ACT), [3, 7, 11, 15])
        self.leg_dofs = model.jnt_dofadr[joints[self.leg_actions]].reshape(4, 3)
        self.q_idx = 1 + model.jnt_qposadr[joints[self.leg_actions]]
        self.dq_idx = 1 + model.nq + model.jnt_dofadr[joints]
        self.dq_scale = np.tile([0.15, 0.15, 0.15, 0.08], 4)
        self.root = 1 + int(model.joint("root").qposadr[0])
        self.root_dof = 1 + model.nq + int(model.joint("root").dofadr[0])
        wheel_geoms = [model.geom(f"{leg}_FOOT_LINK_collision").id for leg in LEGS]
        self.radius, self.half_width = model.geom_size[wheel_geoms[0], :2].copy()
        self.friction_geoms = [model.geom("floor").id, *wheel_geoms]
        self.wheel_sensor = model.sensor(f"{LEGS[0]}_center").adr.item()

        self.home_ctrl = model.key("home").ctrl.copy()
        self.home_leg_q = model.key("home").qpos[self.q_idx - 1].copy()
        self.const_data = data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        mujoco.mj_forward(model, data)
        jac = np.empty((3, model.nv))
        inverse_jac = []
        for k, geom in enumerate(wheel_geoms):
            mujoco.mj_jacGeom(model, data, jac, None, geom)
            j = jac[:, self.leg_dofs[k]]
            inverse_jac.append(j.T @ np.linalg.inv(j @ j.T + 0.015**2 * np.eye(3)))
        self.inverse_jac = np.array(inverse_jac)
        self.foot_xy = data.geom_xpos[wheel_geoms, :2].copy()
        self.home_foot_body = data.geom_xpos[wheel_geoms] - data.qpos[:3]
        self.home_state = np.empty(mujoco.mj_stateSize(model, STATE))
        mujoco.mj_getState(model, data, self.home_state, STATE)
        self.models = [copy.copy(model) for _ in range(num_envs)]
        self.datas = [mujoco.MjData(model) for _ in range(max(nthread, 1))]
        self.roller = rollout.Rollout(nthread=nthread)
        self.state = np.empty((num_envs, self.home_state.size))
        self.trajectory = np.empty((num_envs, DECIMATION, self.home_state.size))
        self.sensors = np.empty((num_envs, DECIMATION, model.nsensordata))
        self.ctrl = np.empty((num_envs, model.nu))
        self.command = np.zeros((num_envs, 3))
        self.action = np.zeros((num_envs, N_ACT), dtype=np.float32)
        self.age = np.zeros(num_envs, dtype=int)
        self.command_left = np.zeros(num_envs, dtype=int)
        self.push_left = np.zeros(num_envs, dtype=int)
        self.tap_remaining = np.zeros(num_envs)
        self.tap_side = np.zeros(num_envs)
        self.tap_heading_error = np.zeros(num_envs)
        self.previous_yaw = np.zeros(num_envs)
        self.phase = np.zeros(num_envs)
        self.lateral = np.zeros(num_envs)
        self.drive_vx = np.zeros(num_envs)
        self.pending_stop = np.zeros(num_envs, dtype=bool)
        self.diagonal = (DiagonalGait(num_envs, self.dt, self.foot_xy, self.cfg.champ_refine)
                         if self.cfg.champ_gait else None)
        self.champ_ik = WheelLegIK(model, data, LEGS, self.home_leg_q, wheel_geoms) if self.cfg.champ_ik else None
        self.reset(np.arange(num_envs))

    def close(self):
        self.roller.close()

    def randomize_models(self, indices, scale=1.0):
        model, base = self.model, self.base
        for i in np.atleast_1d(indices):
            target = self.models[i]
            mass = 1 + scale * self.rng.uniform(-0.12, 0.16)
            target.body_mass[base] = model.body_mass[base] * mass
            target.body_inertia[base] = model.body_inertia[base] * mass
            target.body_ipos[base] = model.body_ipos[base] + scale * self.rng.uniform(-0.018, 0.018, 3)
            target.geom_friction[self.friction_geoms, 0] = 0.9 + scale * self.rng.uniform(-0.45, 0.35)
            kp = 1 + scale * self.rng.uniform(-0.18, 0.18, model.nu)
            kd = 1 + scale * self.rng.uniform(-0.20, 0.20, model.nu)
            target.actuator_gainprm[:, 0] = model.actuator_gainprm[:, 0] * kp
            target.actuator_biasprm[:, 1] = model.actuator_biasprm[:, 1] * kp
            target.actuator_biasprm[:, 2] = model.actuator_biasprm[:, 2] * kd
            target.dof_damping[self.leg_dofs] = scale * self.rng.uniform(0.04, 0.28, (4, 3))
            mujoco.mj_setConst(target, self.const_data)

    def push(self, indices, scale=1.0, azimuth=None):
        ids = np.atleast_1d(indices)
        if azimuth is None:
            azimuth = self.rng.uniform(-np.pi, np.pi, len(ids))
        azimuth = np.broadcast_to(azimuth, (len(ids),))
        speed = scale * self.cfg.push_xy
        dof = self.root_dof
        self.state[ids, dof:dof + 2] += speed * np.column_stack((np.cos(azimuth), np.sin(azimuth)))
        self.state[ids, dof + 2] += scale * self.cfg.push_z * self.rng.choice([-1, 1], len(ids))
        direction = self.rng.normal(size=(len(ids), 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        self.state[ids, dof + 3:dof + 6] += scale * self.cfg.push_angular * direction

    def sample_commands(self, ids):
        stopping = self.pending_stop[ids]
        n = len(ids)
        if self.cfg.vy_turn_only:
            kind = self.rng.choice(4, n, p=[.15, .42, .22, .21])
            commands = np.zeros((n, 3))
            lat = kind == 1
            commands[lat, 1] = self.rng.uniform(.08, self.cfg.max_lateral, lat.sum()) * self.rng.choice([-1, 1], lat.sum())
            slow_turn = kind == 2
            commands[slow_turn, 2] = self.rng.uniform(.14, .32, slow_turn.sum()) * self.rng.choice([-1, 1], slow_turn.sum())
            fast_turn = kind == 3
            commands[fast_turn, 2] = self.rng.uniform(.32, self.cfg.max_yaw, fast_turn.sum()) * self.rng.choice([-1, 1], fast_turn.sum())
            duration = self.rng.uniform(1.5, 2.8, n)
            duration[kind == 0] = self.rng.uniform(1.0, 2.0, (kind == 0).sum())
            duration[slow_turn] = self.rng.uniform(.6, 2.0, slow_turn.sum())
            commands[stopping] = 0
            self.pending_stop[ids] = slow_turn & ~stopping
            self.command[ids] = commands
            self.command_left[ids] = np.maximum(1, np.rint(duration / self.dt)).astype(int)
            return
        mixture = ([.09, .10, .36, .12, .15, .10, .08] if self.cfg.corrective_gait
                   else [.12, .16, .23, .16, .12, .12, .09])
        mode = self.rng.choice(7, n, p=mixture)
        commands = self.rng.uniform(-1, 1, (n, 3)) * [self.cfg.max_speed, self.cfg.max_lateral, self.cfg.max_yaw]
        if self.cfg.max_reverse is not None:
            reverse = commands[:, 0] < 0
            commands[reverse, 0] *= self.cfg.max_reverse / self.cfg.max_speed
        commands[mode == 0] = 0
        commands[mode == 1, 1:] = 0
        commands[mode == 2, [0]] = 0
        commands[mode == 2, [2]] = 0
        if self.cfg.corrective_gait:
            lateral_only = mode == 2
            commands[lateral_only, 1] = (self.rng.uniform(.08, self.cfg.max_lateral, lateral_only.sum())
                                         * self.rng.choice([-1, 1], lateral_only.sum()))
        commands[mode == 3, :2] = 0
        commands[mode == 4, :2] = 0
        if self.cfg.max_speed > .65:
            fast = ((mode == 1) | (mode == 5)) & (self.rng.random(n) < .62)
            commands[fast, 0] = self.cfg.max_speed * self.rng.uniform(.65, 1., fast.sum())
            commands[fast, 1] = 0.
            fast_arc = fast & (mode == 5)
            commands[fast_arc, 2] = self.rng.uniform(.16, .36, fast_arc.sum()) * self.rng.choice([-1, 1], fast_arc.sum())
        duration = self.rng.uniform(1.25, 2.8, n)
        tap = mode == 4
        commands[tap, 2] = self.rng.uniform(.23, self.cfg.max_yaw, tap.sum()) * self.rng.choice([-1, 1], tap.sum())
        duration[tap] = self.rng.uniform(.16, .32, tap.sum())
        commands[stopping] = 0
        fast_stop = (self.cfg.max_speed > .65) & (mode == 1) & (self.rng.random(n) < .25)
        self.pending_stop[ids] = (tap | fast_stop) & ~stopping
        self.command[ids] = commands
        self.command_left[ids] = np.maximum(1, np.rint(duration / self.dt)).astype(int)

    def reset(self, ids):
        self.state[ids] = self.home_state
        self.age[ids] = 0
        self.action[ids] = 0
        self.tap_remaining[ids] = self.tap_side[ids] = self.previous_yaw[ids] = self.lateral[ids] = 0
        self.tap_heading_error[ids] = 0
        self.drive_vx[ids] = 0
        self.phase[ids] = self.rng.choice([0.0, 0.5], len(ids)) if self.training else 0
        self.pending_stop[ids] = False
        if self.diagonal is not None:
            self.reset_gait(ids)
        if self.training:
            self.randomize_models(ids, self.cfg.robustness)
            self.state[np.ix_(ids, self.q_idx)] += self.rng.uniform(-.018, .018, (len(ids), 12))
            self.push_left[ids] = np.rint(self.rng.uniform(2.0, 5.0, len(ids)) / self.dt)
            self.sample_commands(ids)

    def body_state(self):
        root, dof = self.root, self.root_dof
        rotation = quat_to_R(self.state[:, root + 3:root + 7])
        v = np.einsum("nji,nj->ni", rotation, self.state[:, dof:dof + 3])
        return v, self.state[:, dof + 3:dof + 6], -rotation[:, 2, :]

    def measured_yaw(self):
        w, x, y, z = self.state[:, self.root + 3:self.root + 7].T
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y*y + z*z))

    def reset_gait(self, ids):
        if self.diagonal is not None:
            self.diagonal.reset(ids, self.state[ids, self.root:self.root + 2], self.measured_yaw()[ids])

    def observations(self, noise=True):
        v, ang, gravity = self.body_state()
        phase = 2 * np.pi * self.phase
        tap = np.zeros((self.n, 4))
        tap[:, 2] = self.tap_remaining * (self.tap_side > 0) / self.cfg.tap_step_s
        tap[:, 3] = self.tap_remaining * (self.tap_side < 0) / self.cfg.tap_step_s
        if self.diagonal is not None:
            tap = self.diagonal.swing * (self.diagonal.remaining[:, None] / (SUPPORT_S + SWING_S))
        obs = np.column_stack((ang * .3, gravity, self.command, self.state[:, self.q_idx] - self.home_leg_q,
                               self.state[:, self.dq_idx] * self.dq_scale, self.action, np.sin(phase), np.cos(phase),
                               tap, self.tap_heading_error)).astype(np.float32)
        critic = np.column_stack((obs, v, self.state[:, self.root + 2])).astype(np.float32)
        if self.training and noise:
            scale = self.cfg.robustness
            obs[:, :6] += self.rng.normal(0, .009 * scale, (self.n, 6)).astype(np.float32)
            obs[:, 9:37] += self.rng.normal(0, .011 * scale, (self.n, 28)).astype(np.float32)
        return obs, critic

    def controls(self, action):
        if self.diagonal is not None:
            return self.champ_controls(action)
        vx, vy, yaw = self.command.T
        tap = (np.abs(yaw) > .11) & ((np.abs(self.previous_yaw) <= .11) | (yaw * self.previous_yaw < 0))
        tap &= (np.abs(vy) < .045) & (np.abs(vx) < .15)
        self.tap_remaining[tap] = self.cfg.tap_step_s
        self.tap_side[tap] = np.sign(yaw[tap])
        self.tap_heading_error[tap] = 0
        self.previous_yaw[:] = yaw
        self.phase[:] = (self.phase + self.dt * self.cfg.lateral_gait_hz) % 1
        self.lateral += np.clip(vy - self.lateral, -.018, .018)
        phase = (self.phase[:, None] + [0, .5, .5, 0]) % 1
        duty = .66
        swing = np.clip((phase - duty) / (1 - duty), 0, 1)
        s_curve = swing**2 * (3 - 2 * swing)
        stride = 1.85 * self.lateral[:, None] * duty / self.cfg.lateral_gait_hz
        offsets = np.zeros((self.n, 4, 3))
        offsets[:, :, 1] = stride * np.where(phase < duty, .5 - phase / duty, -.5 + s_curve)
        activity = np.clip(np.abs(self.lateral) / .10, 0, 1)
        offsets[:, :, 2] = self.cfg.lateral_lift_m * activity[:, None] * np.sin(np.pi * swing)**2
        if self.cfg.corrective_gait:
            partner_lift = np.sin(np.pi * swing[:, [1, 0, 3, 2]])**2
            offsets[:, :, 2] -= .008 * activity[:, None] * partner_lift

        tap_progress = 1 - self.tap_remaining / self.cfg.tap_step_s
        tap_shape = np.sin(np.pi * tap_progress)**2 * (self.tap_remaining > 0)
        rear = np.column_stack((self.tap_side > 0, self.tap_side < 0))
        tap_lift = self.cfg.lift_m * (1.25 if self.cfg.corrective_gait else 1.65)
        offsets[:, 2:, 2] += tap_lift * tap_shape[:, None] * rear
        offsets[:, 2:, 1] += .023 * (tap_shape * self.tap_side)[:, None] * rear
        self.tap_remaining[:] = np.maximum(0, self.tap_remaining - self.dt)

        position_scale = .17
        wheel_residual = 5 * action[:, 3::4]
        if self.cfg.corrective_gait:
            position_scale = .09 + .13 * activity[:, None]
            wheel_residual = .65 * (wheel_residual - wheel_residual.mean(axis=1, keepdims=True))
        position_residual = position_scale * action[:, self.leg_actions]
        target = self.home_leg_q + np.einsum("kij,nkj->nki", self.inverse_jac, offsets).reshape(-1, 12)
        self.ctrl[:, self.leg_ids] = target + position_residual
        assist = np.clip(4.5 * self.tap_heading_error, -.35, .35)
        if self.cfg.corrective_gait:
            assist += .18 * self.tap_side * tap_shape
        turn = yaw + assist
        wheel = (vx[:, None] + turn[:, None] * [-1.05, 1.05, -1.05, 1.05]) / self.radius
        self.ctrl[:, self.wheel_ids] = np.clip(wheel + wheel_residual, -self.cfg.wheel_speed_limit, self.cfg.wheel_speed_limit)
        return offsets

    def champ_controls(self, action):
        gait = self.diagonal
        _, _, gravity = self.body_state()
        offsets, wheel_turn, error = gait.plan(
            self.command, self.state[:, self.root:self.root + 2], self.measured_yaw(), gravity[:, 1],
            self.state[:, self.q_idx[::3]] - self.home_leg_q[::3],
            (self.cfg.max_lateral, self.cfg.max_yaw),
        )
        # Smooth acceleration/braking ramp prevents wheelies on launch and forward flips on stop.
        delta_vx = self.command[:, 0] - self.drive_vx
        accelerating = (self.command[:, 0] * self.drive_vx >= 0) & (np.abs(self.command[:, 0]) > np.abs(self.drive_vx))
        step_limit = np.where(accelerating, 3.2 * self.dt, 4.2 * self.dt)
        self.drive_vx += np.clip(delta_vx, -step_limit, step_limit)
        self.tap_heading_error[:] = error
        self.tap_remaining[:] = gait.remaining
        self.phase[:] = (gait.pair * .5 + .5 * (1 - gait.remaining / (SUPPORT_S + SWING_S))) % 1
        lift_scale = np.where((gait.remaining > 0) | (np.abs(self.command[:, 1]) > .04) | (np.abs(self.command[:, 2]) > .04), 1., .65)
        target = (self.champ_ik.targets(offsets) if self.champ_ik is not None else
                  self.home_leg_q + np.einsum("kij,nkj->nki", self.inverse_jac, offsets).reshape(-1, 12))
        # Bound learned correction so the reference support sequence remains visible.
        self.ctrl[:, self.leg_ids] = target + .14 * lift_scale[:, None] * action[:, self.leg_actions]
        wheel_action = action[:, 3::4]
        wheel_residual = 1.8 * (wheel_action - wheel_action.mean(axis=1, keepdims=True))
        track = np.where(gait.remaining > 0, .55, .28)[:, None] * [-1., 1., -1., 1.]
        wheel = (self.drive_vx[:, None] + wheel_turn[:, None] * track) / self.radius
        self.ctrl[:, self.wheel_ids] = np.clip(wheel + wheel_residual, -self.cfg.wheel_speed_limit, self.cfg.wheel_speed_limit)
        return offsets

    def step(self, action, *, control_targets=None, external_wrench=None):
        action = np.clip(action, -1, 1)
        if control_targets is None:
            offsets = self.controls(action)
        else:
            self.ctrl[:] = control_targets
            offsets = np.zeros((self.n, 4, 3))
        if self.training and self.cfg.robustness > 0:
            self.push_left -= 1
            ids = np.flatnonzero(self.push_left <= 0)
            if len(ids):
                self.push(ids, self.cfg.robustness)
                self.push_left[ids] = np.rint(self.rng.uniform(2.5, 5.5, len(ids)) / self.dt)
        then = self.state[:, 0].copy()
        control = self.ctrl
        control_spec = mujoco.mjtState.mjSTATE_CTRL
        if external_wrench is not None:
            control_spec |= mujoco.mjtState.mjSTATE_XFRC_APPLIED
            control = np.concatenate((control, external_wrench.reshape(self.n, -1)), axis=1)
        self.roller.rollout(self.models, self.datas, self.state, control[:, None, :],
                            control_spec=control_spec, nstep=DECIMATION,
                            state=self.trajectory, sensordata=self.sensors)
        self.state[:] = self.trajectory[:, -1]
        invalid = ~np.isfinite(self.state).all(axis=1) | (np.abs(self.state[:, 0] - then - self.dt) > .001)
        self.state[invalid] = self.home_state
        v, ang, gravity = self.body_state()
        height = self.state[:, self.root + 2]
        tilt = np.arccos(np.clip(-gravity[:, 2], -1, 1))
        wheel_frames = self.sensors[:, -1, self.wheel_sensor:].reshape(self.n, 4, 7)
        qx, qy = wheel_frames[:, :, 4], wheel_frames[:, :, 5]
        axis_z = np.clip(np.abs(1 - 2 * (qx**2 + qy**2)), 0, 1)
        support = self.radius * np.sqrt(1 - axis_z**2) + self.half_width * axis_z
        clearance = np.maximum(0, np.nan_to_num(wheel_frames[:, :, 2] - support))
        target_vx = self.drive_vx if self.diagonal is not None else self.command[:, 0]
        error = np.column_stack((v[:, 0] - target_vx, v[:, 1] - self.command[:, 1]))
        yaw_error = ang[:, 2] - self.command[:, 2]
        if self.diagonal is None:
            tap_active = (self.tap_remaining > 0) & ~invalid
            self.tap_heading_error[:] = np.where(
                tap_active, np.clip(self.tap_heading_error - self.dt * yaw_error, -.14, .14),
                .86 * self.tap_heading_error,
            )
        terms = {
            "vx": 1.25 * np.exp(-(error[:, 0] / .21)**2),
            "vy": 1.55 * np.exp(-(error[:, 1] / .13)**2)
                  + .70 * np.clip(np.sign(self.command[:, 1]) * v[:, 1] / .23, -.5, 1.0),
            "yaw": 1.15 * np.exp(-(yaw_error / .25)**2),
            "lift": -.25 * np.square((clearance - offsets[:, :, 2]) / .025).mean(axis=1),
            "posture": -2.5 * tilt**2 - 13 * (height - .553)**2,
            "smooth": -.15 * np.square(action - self.action).mean(axis=1) - .09 * np.square(action).mean(axis=1),
        }
        if self.cfg.corrective_gait:
            terms["vy"] += 1.10 * np.clip(np.sign(self.command[:, 1]) * v[:, 1] / .23, -.5, 1.0)
            terms["vx"] -= .6 * np.square(error[:, 0] / .30)
        if self.diagonal is not None:
            swing = self.diagonal.swing if control_targets is None else np.zeros((self.n, 4), dtype=bool)
            rotation = quat_to_R(self.state[:, self.root + 3:self.root + 7])
            base_pos = self.state[:, self.root:self.root + 3]
            foot_body = np.einsum("nji,nkj->nki", rotation, wheel_frames[:, :, :3] - base_pos[:, None, :])
            target_foot_body = self.home_foot_body[None, :, :] + offsets
            foot_err_sq = np.square(foot_body - target_foot_body).sum(axis=2)
            desired_clearance = np.maximum(offsets[:, :, 2], 0.)
            missed_lift = np.maximum(0., desired_clearance * .65 - clearance) * swing
            stance_lift = clearance * (~swing)
            terms["lift"] += (
                1.20 * np.exp(-foot_err_sq.mean(axis=1) / (.035**2))
                - .60 * np.square(missed_lift / .020).sum(axis=1)
                - .80 * np.square(stance_lift / .015).sum(axis=1)
            )
            terms["vy"] = (
                2.00 * np.exp(-(error[:, 1] / .10)**2)
                + 1.20 * np.clip(np.sign(self.command[:, 1]) * v[:, 1] / .23, -.5, 1.0)
            )
            terms["yaw"] += .80 * np.exp(-(self.tap_heading_error / .08)**2)
            zero_vx = np.abs(self.command[:, 0]) < .01
            terms["vx"] -= 1.50 * np.square(v[:, 0]) * zero_vx
            terms["posture"] = -4.0 * tilt**2 - 14.0 * (height - .553)**2 - .50 * np.square(self.tap_heading_error)
        fall = invalid | (height < .37) | (tilt > .68)
        self.age += 1
        timeout = ~fall & (self.age >= self.max_steps)
        done = fall | timeout
        reward = np.clip(sum(terms.values()), 0, None) - fall * 4
        self.action[:] = action
        info = dict(vx=v[:, 0], vy=v[:, 1], yaw=ang[:, 2].copy(), clearance=clearance, tilt=tilt,
                    height=height.copy(), fall=fall, timeout=timeout, vx_error=error[:, 0],
                    vy_error=error[:, 1], yaw_error=yaw_error, terms=terms)
        if self.diagonal is not None:
            info["swing_ref"] = swing.copy()
            info["step_trigger"] = self.diagonal.trigger.copy() if control_targets is None else np.zeros(self.n, dtype=bool)
        if self.training:
            if timeout.any():
                info["terminal_critic"] = self.observations(noise=False)[1][timeout]
            if done.any():
                self.reset(np.flatnonzero(done))
            self.command_left -= 1
            resample = ~done & (self.command_left <= 0)
            if resample.any():
                self.sample_commands(np.flatnonzero(resample))
        return reward.astype(np.float32), done, info
