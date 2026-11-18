"""Go2 速度指令跟踪环境：MuJoCo + 多线程 rollout，向量化（一次推进 N 个环境）。

设计约定（以后 sim2sim 要对齐的东西都在这里）：
- 控制频率 50 Hz（物理步长 0.002 s，每个控制步 10 个物理步）
- 动作 a（12 维，顺序 FL,FR,RL,RR × hip,thigh,calf）：目标关节角 = DEFAULT_Q + ACTION_SCALE * a
- PD 由 MuJoCo 的 position 执行器完成：tau = KP*(target - q) - KD*dq，并截断到力矩上限
- 观测（45 维，actor）：角速度*0.25(3) 投影重力(3) 指令*(2,2,0.25)(3) (q-DEFAULT_Q)(12) dq*0.05(12) 上一个动作(12)
- critic 额外看到特权信息（6 维）：机身系线速度*2(3) 机身高度(1) 最近一次推力的速度增量(2)
- 速度指令 (vx, vy, yaw_rate) 在机身坐标系，vx 范围 [0, vmax]，vmax 由课程从 init_vmax 逐步升到 vx_max
"""
import copy
from dataclasses import asdict, dataclass

import mujoco
import numpy as np
from mujoco import rollout

from rl01.sim import LEGS, SCENE

KP, KD = 20.0, 0.5
ACTION_SCALE = 0.25
DECIMATION = 10
# 顺序 FL, FR, RL, RR；每条腿 hip, thigh, calf（与 unitree_rl_gym 的默认姿态一致）
DEFAULT_Q = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5])
TORQUE_LIMIT = np.array([23.7, 23.7, 45.43] * 4)
N_OBS, N_CRITIC_EXTRA, N_ACT = 45, 6, 12
STAIR_SCAN = 15
STAIR_STAGES = [(h, n) for n in (5, 10, 15) for h in (0.05, 0.10, 0.15)]
STAIR_X0, STAIR_TREAD, STAIR_HALF_WIDTH, STAIR_TOP = 0.8, 0.25, 0.6, 1.0
THIGH_Y = 0.0465 + 0.0955  # 大腿关节在机身系下的横向位置（左 +，右 -）
LEG_SIDE = np.array([1.0, -1.0, 1.0, -1.0])  # FL FR RL RR

# rollout 的状态向量布局（mjSTATE_FULLPHYSICS：time, qpos(19), qvel(18)）
S_Z, S_QUAT, S_Q = 3, slice(4, 8), slice(8, 20)
S_LIN, S_ANG, S_DQ = slice(20, 23), slice(23, 26), slice(26, 38)
NSTATE = 38


@dataclass
class EnvCfg:
    stairs: bool = False
    w_diagonal: float = 0.0  # small moving diagonal-support reward
    stair_front_bonus: float = 0.0  # one-off reward for each newly reached tread
    stair_rear_bonus: float = 0.0
    stair_lateral_limit: float = 0.42  # centre must remain on 1.2 m wide simulated stair
    episode_s: float = 20.0
    # 速度指令
    vx_max: float = 2.0  # 最终的指令上限（m/s）
    init_vmax: float = 0.5  # 课程起点
    vmax_step: float = 0.25
    vy_range: float = 0.0  # 侧向指令范围，默认 0（只练前进）
    yaw_range: float = 0.0
    stand_prob: float = 0.1  # 这部分环境的指令为 0（站立）
    cmd_resample_s: float = 5.0
    # 随机干扰
    push_vel: float = 0.6  # 推力：随机水平速度冲击的最大值（m/s）
    push_interval_s: float = 5.0  # 平均多久被推一次
    friction_range: tuple = (0.4, 1.2)  # 脚摩擦
    mass_range: tuple = (-1.0, 1.5)  # 机身质量增量（kg）
    noise: bool = True  # 观测噪声
    # 终止
    min_height: float = 0.15
    min_cos_tilt: float = 0.5  # 倾角超过 60° 终止
    # 奖励权重（都已乘过 dt 之前的数值）
    track_sigma: float = 0.25  # exp(-squared velocity error / sigma)
    w_track_lin: float = 1.0
    w_track_ang: float = 0.5
    w_lin_vel_z: float = -2.0
    w_ang_vel_xy: float = -0.05
    w_orientation: float = -1.0
    w_torques: float = -2e-4
    w_dof_acc: float = -2.5e-7
    w_action_rate: float = -0.01
    w_feet_air: float = 1.0
    w_collision: float = -1.0
    w_stand_still: float = -0.5
    w_hip: float = -0.1
    w_base_height: float = -50.0
    w_foot_clear: float = -50.0  # 摆动脚离地高度不足的惩罚权重（0 关闭）
    foot_clear_target: float = 0.07  # 摆动脚中心的目标高度（m）；脚中心静止时约 0.011，真实小跑摆动高度约 6-8 cm
    # 低速步态（第 8 次）：摆动脚目标高度随指令速度变化 + 抬过头的上界惩罚 + 低速时额外加重的横滚/俯仰惩罚
    # 全部只在 vx 低于 low_speed_end 时起作用，并在 [low_speed_full, low_speed_end] 之间线性淡出，vx >= low_speed_end 与旧奖励完全一致
    speed_clear: bool = False  # True 才启用下面这些（旧检查点的 env_cfg 里没有这些字段，默认关闭 = 旧行为）
    low_speed_full: float = 0.8  # vx <= 这个值时低速项权重 100%
    low_speed_end: float = 1.0  # vx >= 这个值时低速项权重 0
    w_foot_cap: float = -300.0  # 摆动脚高过 (目标 + cap_margin) 的平方惩罚权重
    cap_margin: float = 0.012
    w_orient_low: float = 0.0  # 额外的 orientation 权重（叠加在 w_orientation 上，乘低速权重）
    w_ang_vel_low: float = 0.0  # 额外的 ang_vel_xy 权重
    w_foot_width: float = -100.0  # 足端相对大腿关节的横向偏差（m）的平方，惩罚外八/内八
    base_height_target: float = 0.28
    air_time_target: float = 0.25
    only_positive_rewards: bool = True

    def as_dict(self):
        return asdict(self)


def build_train_model(visual=False, kp=KP, kd=KD, fast=True, stair_stage=None):
    """训练用模型：执行器改成 PD 位置执行器，加足端和膝盖位置传感器；visual=False 时删掉视觉网格以节省内存。"""
    spec = mujoco.MjSpec.from_file(str(SCENE))
    if not visual:
        for g in list(spec.geoms):
            if g.type == mujoco.mjtGeom.mjGEOM_MESH:
                spec.delete(g)
        for m in list(spec.meshes):
            spec.delete(m)
    if stair_stage is not None:
        h, n = STAIR_STAGES[stair_stage]
        for i in range(15):
            active = i < n
            depth = STAIR_TREAD + (STAIR_TOP if i == n - 1 else 0.0) if active else 0.1
            height = (i + 1) * h if active else 0.05
            x = STAIR_X0 + i * STAIR_TREAD + depth / 2 if active else 100 + i
            spec.worldbody.add_geom(name=f"stair{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[depth / 2, STAIR_HALF_WIDTH, height / 2], pos=[x, 0, height / 2], rgba=[0.75, 0.6, 0.4, 1])
    for i, act in enumerate(spec.actuators):
        lim = float(TORQUE_LIMIT[i])
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.gainprm[0] = kp
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.biasprm[:3] = [0.0, -kp, -kd]
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcerange = [-lim, lim]
    if fast:  # 训练提速：金字塔锥 + impratio=1 + 脚 condim=3 + 少迭代（约 1.6 倍）；sim2sim 时要知道这些差异
        spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        spec.option.impratio = 1
        spec.option.iterations = 30
        spec.option.ls_iterations = 20
        for g in spec.geoms:
            if g.name in LEGS:
                g.condim = 3
    for leg in LEGS:  # 先 4 个脚（传感器 0-11），再 4 个膝盖（12-23）
        spec.add_sensor(name=f"{leg}_foot", type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
                        objtype=mujoco.mjtObj.mjOBJ_GEOM, objname=leg)
    for leg in LEGS:
        spec.add_sensor(name=f"{leg}_knee", type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
                        objtype=mujoco.mjtObj.mjOBJ_BODY, objname=f"{leg}_calf")
    if stair_stage is not None:
        for leg in LEGS:
            spec.body(f"{leg}_calf").add_site(name=f"{leg}_touch_site", type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=[-0.002, 0, -0.213], size=[0.025, 0, 0])
            spec.add_sensor(name=f"{leg}_touch", type=mujoco.mjtSensor.mjSENS_TOUCH, objtype=mujoco.mjtObj.mjOBJ_SITE, objname=f"{leg}_touch_site")
    return spec.compile()


def quat_to_R(q):
    """q: (N,4) w,x,y,z -> (N,3,3)，R 把机身系向量转到世界系。"""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


class Go2Env:
    def __init__(self, num_envs, cfg=None, seed=0, nthread=None, noise=None):
        self.cfg = cfg or EnvCfg()
        if noise is not None:
            self.cfg.noise = noise
        self.N = num_envs
        self.rng = np.random.default_rng(seed)
        self.dt = 0.002 * DECIMATION
        self.max_steps = int(self.cfg.episode_s / self.dt)

        self.stage = 0
        self.env_stage = np.zeros(num_envs, dtype=np.int32)
        self.stage_models = [build_train_model(stair_stage=i) for i in range(len(STAIR_STAGES))] if self.cfg.stairs else []
        base = self.stage_models[0] if self.cfg.stairs else build_train_model()
        self.models = [copy.copy(base) for _ in range(num_envs)]  # 每个环境独立的模型，才能各自随机化摩擦和质量
        self.nthread = nthread or 16
        self.datas = [mujoco.MjData(base) for _ in range(self.nthread)]
        self.roller = rollout.Rollout(nthread=self.nthread)
        self.foot_gid = np.array([base.geom(l).id for l in LEGS])
        self.base_id = base.body("base").id
        self.base_mass = float(base.body_mass[self.base_id])
        self.foot_friction0 = base.geom_friction[self.foot_gid[0]].copy()

        self.state = np.zeros((num_envs, NSTATE))
        self.state_buf = np.zeros((num_envs, DECIMATION, NSTATE))
        self.sens_buf = np.zeros((num_envs, DECIMATION, base.nsensordata))
        # 每个环境的缓存
        self.cmd = np.zeros((num_envs, 3))
        self.cmd_left = np.zeros(num_envs, dtype=np.int64)
        self.last_action = np.zeros((num_envs, N_ACT))
        self.last_dq = np.zeros((num_envs, N_ACT))
        self.air_time = np.zeros((num_envs, 4))
        self.last_contact = np.zeros((num_envs, 4), dtype=bool)
        self.prev_foot = np.zeros((num_envs, 4, 3))  # 上一步足端世界系位置，用来估计足端水平速度
        self.prev_valid = np.zeros(num_envs, dtype=bool)
        self.top_steps = np.zeros(num_envs, dtype=np.int64)
        self.top_touched = np.zeros((num_envs, 4), dtype=bool)
        self.landed_level = np.zeros((num_envs, 4), dtype=np.int32)
        self.landing_candidate = np.zeros((num_envs, 4), dtype=np.int32)
        self.landing_steps = np.zeros((num_envs, 4), dtype=np.int32)
        self.ep_len = np.zeros(num_envs, dtype=np.int64)
        self.ep_return = np.zeros(num_envs)
        self.ep_track = np.zeros(num_envs)
        self.push_delta = np.zeros((num_envs, 2))
        self.vmax = self.cfg.init_vmax
        # 噪声（已乘观测缩放）
        nz = np.zeros(N_OBS)
        nz[0:3], nz[3:6], nz[9:21], nz[21:33] = 0.2 * 0.25, 0.05, 0.01, 1.5 * 0.05
        self.noise_vec = nz
        self._reset_stats()
        self.reset_idx(np.arange(num_envs))
        self._compute_obs()

    # ---------- 统计 ----------
    def _reset_stats(self):
        self.stats = dict(steps=0, swing_z_sum=0.0, swing_n=0, foot_out_sum=0.0, foot_abs_sum=0.0, rel_err_sum=0.0, n_mov=0, tilt2=0.0, rate2=0.0, lo_swing_z=0.0, lo_swing_n=0, track_lin=0.0, track_ang=0.0, vx=0.0, cmd_vx=0.0, falls=0, timeouts=0,
                          ep_return=[], ep_len=[], ep_track=[], stair_success=0, stair_done=0, stair_current_success=0, stair_current_done=0, front_land=0, rear_land=0)

    def pop_stats(self):
        s, self.stats = self.stats, None
        self._reset_stats()
        return s

    def update_curriculum(self, window):
        """window：自上次升级以来累计的 ep_len / rel_err_sum / n_mov。

        条件：≥256 个回合、>85% 的回合跑满、被要求移动的环境平均相对速度误差 < 0.3（站着不动的误差是 1）。
        满足就把指令上限提高一档。
        """
        if len(window["ep_len"]) < 256 or window["n_mov"] < 20000 or self.vmax >= self.cfg.vx_max:
            return False
        survive = np.mean(np.array(window["ep_len"]) >= self.max_steps - 1)
        rel_err = window["rel_err_sum"] / window["n_mov"]
        if rel_err < 0.3 and survive > 0.85:
            self.vmax = min(self.vmax + self.cfg.vmax_step, self.cfg.vx_max)
            return True
        return False

    # ---------- 重置 ----------
    def reset_idx(self, idx):
        if len(idx) == 0:
            return
        n, rng, cfg = len(idx), self.rng, self.cfg
        s = np.zeros((n, NSTATE))
        yaw = rng.uniform(-0.08, 0.08, n) if cfg.stairs else rng.uniform(-np.pi, np.pi, n)
        s[:, 1:3] = 0.0
        if cfg.stairs:
            s[:, 1] = rng.uniform(-0.15, 0.15, n)
            s[:, 2] = rng.uniform(-0.06, 0.06, n)
            self.env_stage[idx] = self.stage
        s[:, S_Z] = 0.35
        s[:, 4], s[:, 7] = np.cos(yaw / 2), np.sin(yaw / 2)
        s[:, S_Q] = DEFAULT_Q + rng.uniform(-0.1, 0.1, (n, 12))
        s[:, 20:22] = rng.uniform(-0.2, 0.2, (n, 2))
        self.state[idx] = s
        # 域随机化：脚摩擦、机身质量
        fr = rng.uniform(*cfg.friction_range, n)
        dm = rng.uniform(*cfg.mass_range, n)
        for k, i in enumerate(idx):
            if cfg.stairs:
                self.models[i] = copy.copy(self.stage_models[self.env_stage[i]])
            m = self.models[i]
            m.geom_friction[self.foot_gid, 0] = fr[k]
            m.body_mass[self.base_id] = self.base_mass + dm[k]
        self.last_action[idx] = 0
        self.last_dq[idx] = 0
        self.air_time[idx] = 0
        self.last_contact[idx] = False
        self.prev_valid[idx] = False  # 重置后第一步没有上一步位置，不算速度
        self.top_steps[idx] = 0
        self.top_touched[idx] = False
        self.landed_level[idx] = 0
        self.landing_candidate[idx] = 0
        self.landing_steps[idx] = 0
        self.ep_len[idx] = 0
        self.ep_return[idx] = 0
        self.ep_track[idx] = 0
        self.push_delta[idx] = 0
        self._resample_cmd(idx)

    def _resample_cmd(self, idx):
        n, rng, cfg = len(idx), self.rng, self.cfg
        vx = rng.uniform(0.25, 0.4, n) if cfg.stairs else rng.uniform(0.0, self.vmax, n)
        if not cfg.stairs:
            vx[rng.random(n) < cfg.stand_prob] = 0.0
        self.cmd[idx, 0] = vx
        self.cmd[idx, 1] = rng.uniform(-cfg.vy_range, cfg.vy_range, n)
        self.cmd[idx, 2] = rng.uniform(-cfg.yaw_range, cfg.yaw_range, n)
        steps = int(cfg.cmd_resample_s / self.dt)
        self.cmd_left[idx] = rng.integers(steps // 2, steps + 1, n)

    # ---------- 观测 ----------
    def _unpack(self):
        s = self.state
        R = quat_to_R(s[:, S_QUAT])
        self.R = R
        self.lin_b = np.einsum("nji,nj->ni", R, s[:, S_LIN])  # 机身系线速度
        self.ang_b = s[:, S_ANG]  # free joint 的角速度本来就在机身系
        self.g_b = -R[:, 2, :]  # 机身系下的重力方向
        self.q, self.dq = s[:, S_Q], s[:, S_DQ]

    def _compute_obs(self):
        self._unpack()
        scale = np.array([2.0, 2.0, 0.25])
        obs = np.concatenate([self.ang_b * 0.25, self.g_b, self.cmd * scale, self.q - DEFAULT_Q,
                              self.dq * 0.05, self.last_action], axis=1)
        self.critic_extra = np.concatenate([self.lin_b * 2.0, self.state[:, S_Z : S_Z + 1], self.push_delta], axis=1)
        if self.cfg.stairs:
            dx = np.repeat(np.array([-0.3, 0.0, 0.3, 0.6, 0.9]), 3)
            dy = np.tile(np.array([-0.25, 0.0, 0.25]), 5)
            x = self.state[:, 1, None] + self.R[:, 0, 0, None] * dx + self.R[:, 0, 1, None] * dy
            y = self.state[:, 2, None] + self.R[:, 1, 0, None] * dx + self.R[:, 1, 1, None] * dy
            scan = self.terrain_height(x, y) - self.state[:, S_Z, None]
            self.critic_extra = np.concatenate([self.critic_extra, scan], axis=1)
        self.obs_clean = obs
        if self.cfg.noise:
            obs = obs + self.rng.uniform(-1, 1, obs.shape) * self.noise_vec
        self.obs = obs

    def get_obs(self):
        return self.obs.astype(np.float32), np.concatenate([self.obs_clean, self.critic_extra], 1).astype(np.float32)

    def terrain_height(self, x, y):
        """Analytic top surface; not a contact sensor near a vertical riser."""
        h = np.array([STAIR_STAGES[i][0] for i in self.env_stage])[:, None]
        n = np.array([STAIR_STAGES[i][1] for i in self.env_stage])[:, None]
        level = np.clip(np.floor((x - STAIR_X0) / STAIR_TREAD) + 1, 0, n)
        inside = (np.abs(y) <= STAIR_HALF_WIDTH) & (x < STAIR_X0 + (n - 1) * STAIR_TREAD + STAIR_TREAD + STAIR_TOP)
        return np.where(inside, level * h, 0.0)

    def stair_landing_reward(self, feet, touch):
        """One reward per foot and newly reached tread; rear follows a front foot.

        Touch must persist 3 control steps and the foot centre be near the tread
        surface, preventing wall contacts and repeated hops from farming reward.
        """
        h = np.array([STAIR_STAGES[i][0] for i in self.env_stage])[:, None]
        n = np.array([STAIR_STAGES[i][1] for i in self.env_stage])[:, None]
        level = np.clip(np.floor((feet[:, :, 0] - STAIR_X0) / STAIR_TREAD).astype(int) + 1, 0, n)
        valid = (touch & (level > 0) & (np.abs(feet[:, :, 1]) < STAIR_HALF_WIDTH) &
                 (np.abs(feet[:, :, 2] - (level * h + 0.022)) < 0.025))
        level = np.where(valid, level, 0)
        self.landing_steps = np.where(level > 0, np.where(level == self.landing_candidate, self.landing_steps + 1, 1), 0)
        self.landing_candidate = level
        new = (level > self.landed_level) & (self.landing_steps >= 3)
        front = new[:, :2]
        self.landed_level[:, :2] = np.where(front, level[:, :2], self.landed_level[:, :2])
        rear = new[:, 2:] & (level[:, 2:] <= self.landed_level[:, :2].max(axis=1, keepdims=True))
        self.landed_level[:, 2:] = np.where(rear, level[:, 2:], self.landed_level[:, 2:])
        self.front_land_events = int(front.sum())
        self.rear_land_events = int(rear.sum())
        return front.sum(axis=1) * self.cfg.stair_front_bonus + rear.sum(axis=1) * self.cfg.stair_rear_bonus

    def stair_top_success(self, fw, contact, fall):
        ns = np.array([STAIR_STAGES[i][1] for i in self.env_stage])
        hs = np.array([STAIR_STAGES[i][0] for i in self.env_stage])
        edge = STAIR_X0 + (ns - 1) * STAIR_TREAD
        far_edge = edge + STAIR_TREAD + STAIR_TOP
        on_top = ((fw[:, :, 0] > edge[:, None] + 0.06).all(1) &
                  (fw[:, :, 0] < far_edge[:, None] - 0.03).all(1) &
                  (fw[:, :, 2] > (ns * hs - 0.04)[:, None]).all(1) &
                  (np.abs(fw[:, :, 1]) < STAIR_HALF_WIDTH).all(1) &
                  (self.state[:, 1] > edge + 0.08) & ~fall)
        # A trot lifts alternating feet; require each to touch the top at least
        # once during a continuous half-second, not all four at once.
        top_contact = contact & (np.abs(fw[:, :, 2] - (ns * hs + 0.022)[:, None]) < 0.04)
        self.top_touched = np.where(on_top[:, None], self.top_touched | top_contact, False)
        self.top_steps = np.where(on_top, self.top_steps + 1, 0)
        return (self.top_steps >= 25) & self.top_touched.all(axis=1)

    # ---------- 一步 ----------
    def step(self, action):
        cfg, dt, N = self.cfg, self.dt, self.N
        action = np.clip(action, -10.0, 10.0)
        target = DEFAULT_Q + ACTION_SCALE * action

        # 随机推力：给机身一个水平速度冲击
        self.push_delta[:] = 0
        if cfg.push_vel > 0:
            m = self.rng.random(N) < dt / cfg.push_interval_s
            d = self.rng.uniform(-cfg.push_vel, cfg.push_vel, (N, 2))
            self.state[m, 20:22] += d[m]
            self.push_delta[m] = d[m]

        self.roller.rollout(self.models, self.datas, self.state, target[:, None, :], nstep=DECIMATION,
                            state=self.state_buf, sensordata=self.sens_buf)
        self.state = self.state_buf[:, -1].copy()
        sens = self.sens_buf[:, -1]
        foot_z = sens[:, 2:12:3]
        knee_z = sens[:, 14:24:3]
        if cfg.stairs:
            fw = sens[:, :12].reshape(N, 4, 3)
            kw = sens[:, 12:24].reshape(N, 4, 3)
            hs = np.array([STAIR_STAGES[i][0] for i in self.env_stage])[:, None]
            # If a point is below the upper tread at a riser, it cannot be supported by that tread.
            fground = np.minimum(self.terrain_height(fw[:, :, 0], fw[:, :, 1]), np.maximum(0, np.floor(fw[:, :, 2] / hs) * hs))
            kground = np.minimum(self.terrain_height(kw[:, :, 0], kw[:, :, 1]), np.maximum(0, np.floor(kw[:, :, 2] / hs) * hs))
            foot_z = foot_z - fground
            knee_z = knee_z - kground
            body_ground = fground.mean(axis=1)
        bad = ~np.isfinite(self.state).all(1) | (np.abs(self.state[:, 20:38]) > 1e3).any(1)
        if bad.any():
            self.state[bad] = 0.0
            self.state[bad, 3], self.state[bad, 4] = 0.35, 1.0
            self.state[bad, S_Q] = DEFAULT_Q

        self._unpack()
        # ---- 奖励 ----
        lin_err = np.sum((self.cmd[:, :2] - self.lin_b[:, :2]) ** 2, 1)
        ang_err = (self.cmd[:, 2] - self.ang_b[:, 2]) ** 2
        track_lin = np.exp(-lin_err / cfg.track_sigma)
        track_ang = np.exp(-ang_err / 0.25)
        tau = np.clip(KP * (target - self.q) - KD * self.dq, -TORQUE_LIMIT, TORQUE_LIMIT)
        contact = sens[:, 24:28] > 0.1 if cfg.stairs else foot_z < 0.03
        landing_reward = self.stair_landing_reward(fw, contact) if cfg.stairs else np.zeros(N)
        contact_f = contact | self.last_contact
        first = (self.air_time > 0.0) & contact_f
        self.air_time += dt
        moving = np.linalg.norm(self.cmd[:, :2], axis=1) > 0.1
        feet_air = np.sum((self.air_time - cfg.air_time_target) * first, 1) * moving
        self.air_time *= ~contact_f
        self.last_contact = contact
        standing = np.linalg.norm(self.cmd, axis=1) < 0.1
        hip_idx = [0, 3, 6, 9]
        foot_w = sens[:, :12].reshape(N, 4, 3)
        foot_b = np.einsum("nji,nkj->nki", self.R, foot_w - self.state[:, None, 1:4])  # 足端在机身系下的位置
        out_y = LEG_SIDE * foot_b[:, :, 1] - THIGH_Y  # 正 = 比大腿关节更靠外（外八）
        foot_speed = np.linalg.norm(foot_w[:, :, :2] - self.prev_foot[:, :, :2], axis=-1) / dt * self.prev_valid[:, None]
        self.prev_foot = foot_w.copy()  # 必须 copy：foot_w 来自 sens_buf 的视图，下一步会被覆盖
        self.prev_valid[:] = True  # 之后 reset_idx 会把终止的环境置回 False
        swing = foot_speed > 0.5  # 脚在快速移动 = 摆动相
        clear_pen = np.maximum(cfg.foot_clear_target - foot_z, 0.0) ** 2 * np.tanh(2.0 * foot_speed)
        low_w = np.clip((cfg.low_speed_end - self.cmd[:, 0]) / (cfg.low_speed_end - cfg.low_speed_full), 0.0, 1.0) if cfg.speed_clear else np.zeros(N)
        if cfg.speed_clear:  # 目标高度随指令速度：0.3 以下 3 cm，0.5 → 4 cm，0.8 → 5.5 cm，1.0 以上 7 cm（线性插值）
            tgt = np.interp(self.cmd[:, 0], [0.3, 0.5, 0.8, 1.0], [0.03, 0.04, 0.055, cfg.foot_clear_target])[:, None]
            tgt = np.where(self.cmd[:, 0:1] >= 1.0, cfg.foot_clear_target, tgt)
            # 按相对不足量放大：目标越低权重越大（(0.07/目标)^2，0.3 以下约 5.4 倍），>=1.0 m/s 恒为 1，与旧奖励一致。
            # 不放大时 3 cm 的目标几乎没有惩罚，v6 第 150 步低速退回了蹭地（触地 100%，抬脚 <3 cm）
            clear_pen = np.maximum(tgt - foot_z, 0.0) ** 2 * np.tanh(2.0 * foot_speed) * (cfg.foot_clear_target / tgt) ** 2
            cap_pen = np.maximum(foot_z - (tgt + cfg.cap_margin), 0.0) ** 2 * np.tanh(2.0 * foot_speed)
        else:
            cap_pen = np.zeros_like(clear_pen)
        diagonal = ((contact[:, 0] & contact[:, 3] & ~contact[:, 1] & ~contact[:, 2]) |
                    (contact[:, 1] & contact[:, 2] & ~contact[:, 0] & ~contact[:, 3])) if cfg.stairs else np.zeros(N, dtype=bool)
        diagonal *= self.lin_b[:, 0] > 0.08  # no reward for freezing in a diagonal pose
        terms = dict(
            diagonal=cfg.w_diagonal * diagonal,
            stair_landing=landing_reward / dt,
            track_lin=cfg.w_track_lin * track_lin,
            track_ang=cfg.w_track_ang * track_ang,
            lin_vel_z=cfg.w_lin_vel_z * self.lin_b[:, 2] ** 2,
            ang_vel_xy=(cfg.w_ang_vel_xy + cfg.w_ang_vel_low * low_w) * np.sum(self.ang_b[:, :2] ** 2, 1),
            orientation=(cfg.w_orientation + cfg.w_orient_low * low_w) * np.sum(self.g_b[:, :2] ** 2, 1),
            torques=cfg.w_torques * np.sum(tau ** 2, 1),
            dof_acc=cfg.w_dof_acc * np.sum(((self.dq - self.last_dq) / dt) ** 2, 1),
            action_rate=cfg.w_action_rate * np.sum((action - self.last_action) ** 2, 1),
            feet_air=cfg.w_feet_air * feet_air,
            collision=cfg.w_collision * np.sum(knee_z < 0.035, 1),
            stand_still=cfg.w_stand_still * np.sum(np.abs(self.q - DEFAULT_Q), 1) * standing,
            hip=cfg.w_hip * np.sum((self.q[:, hip_idx] - DEFAULT_Q[hip_idx]) ** 2, 1),
            base_height=cfg.w_base_height * (self.state[:, S_Z] - (body_ground if cfg.stairs else 0.0) - cfg.base_height_target) ** 2,
            foot_width=cfg.w_foot_width * np.mean(out_y ** 2, 1),
            foot_clear=cfg.w_foot_clear * np.mean(clear_pen, 1),
            foot_cap=cfg.w_foot_cap * np.mean(cap_pen, 1) * low_w,
        )
        rew = sum(terms.values()) * dt
        self.term_means = {k: float(v.mean() * dt) for k, v in terms.items()}  # 调试：每项每步的平均贡献
        if cfg.only_positive_rewards:  # 总奖励为负时"早点摔倒"反而划算，所以截断在 0 以上（legged_gym 的做法）
            rew = np.maximum(rew, 0.0)
        self.last_action = action.copy()
        self.last_dq = self.dq.copy()

        # ---- 终止 ----
        self.ep_len += 1
        ground = body_ground if cfg.stairs else 0.0
        fall = bad | (self.state[:, S_Z] - ground < cfg.min_height) | (-self.g_b[:, 2] < cfg.min_cos_tilt)
        if cfg.stairs:
            fall |= np.abs(self.state[:, 2]) > cfg.stair_lateral_limit
        success = np.zeros(N, dtype=bool)
        if cfg.stairs:
            success = self.stair_top_success(fw, contact, fall)
            rew += success.astype(float) * 2.0
        timeout = (self.ep_len >= self.max_steps) & ~fall & ~success
        done = fall | timeout | success
        self.last_outcome = (success.copy(), fall.copy(), timeout.copy())
        self.ep_return += rew
        self.ep_track += track_lin

        st = self.stats
        st["steps"] += N
        st["swing_z_sum"] += float(foot_z[swing].sum())
        st["swing_n"] += int(swing.sum())
        st["foot_out_sum"] += float(out_y.mean(1).sum())
        st["foot_abs_sum"] += float(np.abs(out_y).mean(1).sum())
        mv = self.cmd[:, 0] > 0.3  # 只看"被要求移动"的环境：相对速度误差 |cmd-vx|/cmd，站着不动 = 1
        st["rel_err_sum"] += float((np.abs(self.cmd[mv, 0] - self.lin_b[mv, 0]) / self.cmd[mv, 0]).sum())
        st["n_mov"] += int(mv.sum())
        st["tilt2"] += float(np.sum(self.g_b[:, :2] ** 2))
        st["rate2"] += float(np.sum(self.ang_b[:, :2] ** 2))
        lo = (self.cmd[:, 0] > 0.1) & (self.cmd[:, 0] < 0.8)  # 低速（要求移动）环境的摆动脚高度
        lsw = swing & lo[:, None]
        st["lo_swing_z"] += float(foot_z[lsw].sum())
        st["lo_swing_n"] += int(lsw.sum())
        st["track_lin"] += track_lin.sum()
        st["track_ang"] += track_ang.sum()
        st["vx"] += self.lin_b[:, 0].sum()
        st["cmd_vx"] += self.cmd[:, 0].sum()
        if cfg.stairs:
            st["front_land"] += self.front_land_events
            st["rear_land"] += self.rear_land_events
        st["stair_success"] += int(success.sum())
        st["stair_done"] += int(done.sum())
        st["stair_current_success"] += int((success & (self.env_stage == self.stage)).sum())
        st["stair_current_done"] += int((done & (self.env_stage == self.stage)).sum())
        st["falls"] += int(fall.sum())
        st["timeouts"] += int(timeout.sum())
        di = np.nonzero(done)[0]
        if len(di):
            st["ep_return"] += self.ep_return[di].tolist()
            st["ep_len"] += self.ep_len[di].tolist()
            st["ep_track"] += self.ep_track[di].tolist()
        self.reset_idx(di)
        # 没有终止的环境到时间就重新采样指令
        self.cmd_left -= 1
        ri = np.nonzero((self.cmd_left <= 0) & ~done)[0]
        if len(ri):
            self._resample_cmd(ri)
        self._compute_obs()
        return rew, done, timeout
