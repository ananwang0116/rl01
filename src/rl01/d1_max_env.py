"""D1 Max 平地轮式速度跟踪。策略动作是站姿和差速轮速前馈的残差。"""
import copy
from dataclasses import dataclass

import mujoco
from mujoco import rollout
import numpy as np

from rl01.env import quat_to_R
from rl01.sim import D1_MAX_SCENE

LEGS = ("FBL", "FAR", "RBL", "RAR")
JOINT_NAMES = tuple(
    f"{leg}_{joint}_JOINT" for leg in LEGS for joint in ("ABAD", "HIP", "KNEE", "FOOT")
)
N_ACT = len(JOINT_NAMES)
N_OBS, N_CRITIC_OBS = 52, 56
DECIMATION = 10
STATE = mujoco.mjtState.mjSTATE_FULLPHYSICS
# y 反射时，x 轴转动换号，y 轴转动保持同号。
JOINT_MIRROR = [4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11]
JOINT_SIGNS = (-1, 1, 1, 1) * 4
LEG_MIRROR = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]


@dataclass(frozen=True)
class FlatConfig:
    max_speed: float = 0.8
    max_yaw: float = 0.6
    episode_s: float = 8.0
    command_s: float = 2.0
    friction_range: tuple = (0.8, 1.05)
    leg_residual_rad: float = 0.22
    wheel_residual_rad_s: float = 6.0
    wheel_target_limit_rad_s: float = 14.0
    # Skid steering needs extra differential wheel speed to overcome lateral scrub.
    turn_feedforward_m: float = 1.15
    full_turn_residual_yaw: float = 0.0


def mirror_action(action):
    return action[..., JOINT_MIRROR] * action.new_tensor(JOINT_SIGNS)


def mirror_obs(obs):
    mirrored = obs.clone()
    mirrored[..., :8] *= obs.new_tensor((-1, 1, -1, 1, -1, 1, 1, -1))
    mirrored[..., 8:20] = obs[..., 8:20][..., LEG_MIRROR] * obs.new_tensor((-1, 1, 1) * 4)
    mirrored[..., 20:36] = mirror_action(obs[..., 20:36])
    mirrored[..., 36:52] = mirror_action(obs[..., 36:52])
    return mirrored


def build_flat_model(visual=False):
    spec = mujoco.MjSpec.from_file(str(D1_MAX_SCENE))
    if not visual:
        # Meshes in this MJCF are visual only. Keep every body, joint and collision geom.
        for geom in list(spec.geoms):
            if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
                spec.delete(geom)
        for mesh in list(spec.meshes):
            spec.delete(mesh)
    return spec.compile()


class D1MaxFlatEnv:
    def __init__(self, num_envs, cfg=None, *, seed=0, nthread=0, fixed_command=None, visual=False):
        self.cfg = cfg or FlatConfig()
        self.n = num_envs
        self.rng = np.random.default_rng(seed)
        self.fixed_command = fixed_command
        self.model = build_flat_model(visual)
        model = self.model
        self.dt = model.opt.timestep * DECIMATION
        self.max_steps = max(1, round(self.cfg.episode_s / self.dt))
        self.command_steps = max(1, round(self.cfg.command_s / self.dt))
        self.act_ids = np.array([model.actuator(name).id for name in JOINT_NAMES])
        joints = np.array([model.joint(name).id for name in JOINT_NAMES])
        if not np.array_equal(model.actuator_trnid[self.act_ids, 0], joints):
            raise ValueError("D1 Max actuator and joint names must match")
        self.wheel = np.array([name.endswith("_FOOT_JOINT") for name in JOINT_NAMES])
        self.side = np.array([-1.0, 1.0, -1.0, 1.0])
        self.leg_ids = self.act_ids[~self.wheel]
        self.wheel_ids = self.act_ids[self.wheel]
        self.q_idx = 1 + model.jnt_qposadr[joints[~self.wheel]]
        self.dq_idx = 1 + model.nq + model.jnt_dofadr[joints]
        self.root = int(model.joint("root").qposadr[0]) + 1
        self.root_dof = 1 + model.nq + int(model.joint("root").dofadr[0])
        wheel_geoms = [model.geom(f"{leg}_FOOT_LINK_collision").id for leg in LEGS]
        self.wheel_radius = float(model.geom_size[wheel_geoms[0], 0])
        self.friction_geoms = [model.geom("floor").id, *wheel_geoms]
        self.home_ctrl = model.key("home").ctrl.copy()
        self.home_leg_q = model.key("home").qpos[self.q_idx - 1].copy()
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        self.home_state = np.empty(mujoco.mj_stateSize(model, STATE))
        mujoco.mj_getState(model, data, self.home_state, STATE)
        self.randomize = fixed_command is None
        self.models = [copy.copy(model) for _ in range(num_envs)] if self.randomize else model
        self.datas = [mujoco.MjData(model) for _ in range(max(1, nthread))]
        self.roller = rollout.Rollout(nthread=nthread)
        self.state = np.empty((num_envs, self.home_state.size))
        self.trajectory = np.empty((num_envs, DECIMATION, self.home_state.size))
        self.sensors = np.empty((num_envs, DECIMATION, model.nsensordata))
        self.ctrl = np.empty((num_envs, model.nu))
        self.command = np.empty((num_envs, 2))
        self.action = np.zeros((num_envs, N_ACT), dtype=np.float32)
        self.age = np.zeros(num_envs, dtype=int)
        self.command_left = np.zeros(num_envs, dtype=int)
        self.reset(np.arange(num_envs))

    def close(self):
        self.roller.close()

    def sample_command(self, ids):
        if self.fixed_command is not None:
            self.command[ids] = self.fixed_command
            self.command_left[ids] = self.command_steps
        else:
            limits = [self.cfg.max_speed, self.cfg.max_yaw]
            self.command[ids] = self.rng.uniform(-1, 1, (len(ids), 2)) * limits
            kind = self.rng.random(len(ids))
            self.command[ids[kind < 0.30], 0] = 0  # stationary turns
            self.command[ids[kind < 0.08]] = 0
            self.command[ids[kind > 0.90], 1] = 0  # straight
            duration_scale = self.rng.uniform(0.75, 1.25, len(ids))
            self.command_left[ids] = np.maximum(1, np.rint(self.command_steps * duration_scale))

    def reset(self, ids):
        self.state[ids] = self.home_state
        if self.randomize:
            noise = self.rng.uniform(-0.025, 0.025, (len(ids), self.q_idx.size))
            self.state[np.ix_(ids, self.q_idx)] += noise
            for idx, friction in zip(ids, self.rng.uniform(*self.cfg.friction_range, len(ids))):
                self.models[idx].geom_friction[self.friction_geoms, 0] = friction
        self.age[ids] = 0
        self.action[ids] = 0
        self.sample_command(ids)

    def body_state(self):
        root, dof = self.root, self.root_dof
        rotation = quat_to_R(self.state[:, root + 3:root + 7])
        body_velocity = np.einsum("nji,nj->ni", rotation, self.state[:, dof:dof + 3])
        return body_velocity, self.state[:, dof + 3:dof + 6], -rotation[:, 2, :]

    def observations(self):
        local_velocity, angular_velocity, gravity = self.body_state()
        joint_velocity = self.state[:, self.dq_idx] * np.where(self.wheel, 0.08, 0.15)
        obs = np.column_stack((
            angular_velocity * 0.3, gravity, self.command,
            self.state[:, self.q_idx] - self.home_leg_q, joint_velocity, self.action,
        )).astype(np.float32)
        critic = np.column_stack((obs, local_velocity, self.state[:, self.root + 2])).astype(np.float32)
        return obs, critic

    def control(self, action):
        self.ctrl[:] = self.home_ctrl
        yaw = self.command[:, 1:2]
        if self.cfg.full_turn_residual_yaw > 0:
            action = action * np.minimum(np.abs(yaw) / self.cfg.full_turn_residual_yaw, 1)
        self.ctrl[:, self.leg_ids] += self.cfg.leg_residual_rad * action[:, ~self.wheel]
        drive = (self.command[:, :1] +
                 yaw * self.side * self.cfg.turn_feedforward_m) / self.wheel_radius
        drive += self.cfg.wheel_residual_rad_s * action[:, self.wheel]
        limit = self.cfg.wheel_target_limit_rad_s
        self.ctrl[:, self.wheel_ids] = np.clip(drive, -limit, limit)
        return self.ctrl

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        control = self.control(action)
        prev_time = self.state[:, 0].copy()
        self.roller.rollout(
            self.models, self.datas, self.state, control[:, None, :], nstep=DECIMATION,
            state=self.trajectory, sensordata=self.sensors,
        )
        self.state[:] = self.trajectory[:, -1]
        invalid = (~np.isfinite(self.state).all(axis=1) |
                   (np.abs(self.state[:, 0] - prev_time - self.dt) > 0.001))
        self.state[invalid] = self.home_state
        self.age += 1
        self.command_left -= 1

        linear, angular, gravity = self.body_state()
        vx_error = linear[:, 0] - self.command[:, 0]
        yaw_error = angular[:, 2] - self.command[:, 1]
        height = self.state[:, self.root + 2]
        reward = (1.5 * np.exp(-(vx_error / 0.24)**2) +
                  1.2 * np.exp(-(yaw_error / 0.28)**2) -
                  1.2 * np.square(linear[:, 1]) -
                  0.4 * np.square(gravity[:, :2]).sum(axis=1) -
                  0.12 * np.square(angular[:, :2]).sum(axis=1) -
                  0.8 * np.square(self.state[:, self.q_idx] - self.home_leg_q).mean(axis=1) -
                  0.07 * np.square(action).mean(axis=1) -
                  0.05 * np.square(action - self.action).mean(axis=1))
        fall = invalid | (height < 0.38) | (gravity[:, 2] > -0.7)
        timeout = ~fall & (self.age >= self.max_steps)
        done = fall | timeout
        reward[fall] -= 4.0
        self.action[:] = action
        info = {
            "vx": linear[:, 0], "vy": linear[:, 1], "yaw": angular[:, 2],
            "vx_error": vx_error, "yaw_error": yaw_error, "fall": fall,
            "timeout": timeout,
        }
        if timeout.any():
            info["terminal_critic"] = self.observations()[1][timeout]
        if done.any():
            self.reset(np.flatnonzero(done))
        resample = ~done & (self.command_left == 0)
        if resample.any():
            self.sample_command(np.flatnonzero(resample))
        return reward.astype(np.float32), done, info
