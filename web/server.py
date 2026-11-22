"""WebLab 控制台后端：为 rl01 (Unitree Go2) 提供单端口 HTTP + WebSocket 实时仿真服务。

启动方式：
    uv run python web/server.py --port 8800
打开浏览器访问：
    http://127.0.0.1:8800/
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import struct
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import mujoco
import numpy as np
import torch
from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.http11 import Response

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rl01.env import ACTION_SCALE, DECIMATION, DEFAULT_Q, KD, KP, N_OBS, TORQUE_LIMIT  # noqa: E402
from rl01.play import load_policy  # noqa: E402
from rl01.sim import LEGS, SCENE  # noqa: E402

JOINT_NAMES = [
    f"{leg}_{part}_joint"
    for leg in LEGS
    for part in ("hip", "thigh", "calf")
]

POSTURE_PRESETS = {
    "stand": DEFAULT_Q.copy(),
    "home": np.array([0.0, 0.9, -1.8] * 4, dtype=np.float64),
    "crouch": np.array([0.0, 1.35, -2.4] * 4, dtype=np.float64),
    "lie": np.array([0.0, 1.58, -2.68] * 4, dtype=np.float64),
    "zero": np.zeros(12, dtype=np.float64),
}

STAIR_X0 = 0.8
STAIR_HALF_WIDTH = 0.65
STAIR_TOP_LEN = 1.2


def build_sim_model(step_h: float = 0.0, n_steps: int = 10, tread: float = 0.25) -> tuple[mujoco.MjModel, list[dict]]:
    """构建带可视化网格与 PD 执行器的 Go2 MuJoCo 模型，并记录楼梯几何描述。"""
    spec = mujoco.MjSpec.from_file(str(SCENE))
    stairs_meta: list[dict] = []
    if step_h > 0 and n_steps > 0:
        for i in range(n_steps):
            depth = tread + (STAIR_TOP_LEN if i == n_steps - 1 else 0.0)
            height = (i + 1) * step_h
            cx = STAIR_X0 + i * tread + depth / 2.0
            spec.worldbody.add_geom(
                name=f"stair{i}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[depth / 2.0, STAIR_HALF_WIDTH, height / 2.0],
                pos=[cx, 0.0, height / 2.0],
                rgba=[0.76, 0.63, 0.45, 1.0],
            )
            stairs_meta.append({
                "pos": [cx, 0.0, height / 2.0],
                "size": [depth / 2.0, STAIR_HALF_WIDTH, height / 2.0],
                "rgba": [0.76, 0.63, 0.45, 1.0],
            })

    for i, act in enumerate(spec.actuators):
        lim = float(TORQUE_LIMIT[i])
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.gainprm[0] = KP
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.biasprm[:3] = [0.0, -KP, -KD]
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcerange = [-lim, lim]

    model = spec.compile()
    return model, stairs_meta


def pack_meshes_binary(model: mujoco.MjModel) -> bytes:
    """将 MuJoCo 编译后的视觉网格（位置+平滑法线+三角面索引）打包为二进制流。"""
    chunks = [struct.pack("<I", model.nmesh)]
    for i in range(model.nmesh):
        vadr = int(model.mesh_vertadr[i])
        vnum = int(model.mesh_vertnum[i])
        fadr = int(model.mesh_faceadr[i])
        fnum = int(model.mesh_facenum[i])

        verts = np.asarray(model.mesh_vert[vadr : vadr + vnum], dtype=np.float32)
        faces = np.asarray(model.mesh_face[fadr : fadr + fnum], dtype=np.uint32)

        # 用面法线累加计算平滑顶点法线，避免 mesh_normal 与 mesh_vert 索引不一致
        normals = np.zeros_like(verts, dtype=np.float32)
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        np.add.at(normals, faces[:, 0], fn)
        np.add.at(normals, faces[:, 1], fn)
        np.add.at(normals, faces[:, 2], fn)
        lens = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(normals, np.maximum(lens, 1e-8))

        chunks.append(struct.pack("<II", vnum, fnum * 3))
        chunks.append(verts.tobytes())
        chunks.append(normals.tobytes())
        chunks.append(faces.tobytes())
    return b"".join(chunks)


def extract_robot_geoms(model: mujoco.MjModel) -> list[dict]:
    """提取机器人自身各连杆的视觉/碰撞几何体描述（排除地板与楼梯）。"""
    geoms = []
    for i in range(model.ngeom):
        body_id = int(model.geom_bodyid[i])
        if body_id == 0:
            continue
        mid = int(model.geom_matid[i])
        rgba = (
            model.mat_rgba[mid].tolist()
            if mid >= 0
            else model.geom_rgba[i].tolist()
        )
        geoms.append({
            "id": i,
            "name": model.geom(i).name or "",
            "body_id": body_id,
            "type": int(model.geom_type[i]),
            "group": int(model.geom_group[i]),
            "mesh_id": int(model.geom_dataid[i]),
            "size": model.geom_size[i].tolist(),
            "pos": model.geom_pos[i].tolist(),
            "quat": model.geom_quat[i].tolist(),
            "rgba": rgba,
        })
    return geoms


def scan_checkpoints() -> list[str]:
    runs_dir = ROOT / "runs"
    if not runs_dir.exists():
        return []
    pts = sorted(runs_dir.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p.relative_to(ROOT)) for p in pts]


def scan_logs() -> list[str]:
    runs_dir = ROOT / "runs"
    if not runs_dir.exists():
        return []
    logs = sorted(runs_dir.glob("train_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in logs]


def parse_training_log(filename: str) -> dict:
    path = (ROOT / "runs" / filename).resolve()
    if not path.exists() or path.parent != (ROOT / "runs").resolve():
        return {"error": "Log not found"}

    text = path.read_text(encoding="utf-8", errors="ignore")
    iters, ep_ret, track_lin, vx, cmd_vx, falls, survive = [], [], [], [], [], [], []
    cur: dict | None = None
    re_iter = re.compile(r"^\[\s*(\d+)/\d+")
    re_ret = re.compile(r"回合回报\s+([+-]?\d+(?:\.\d+)?).*?存活\s+(\d+)/(\d+).*?摔倒\s+(\d+)")
    re_vel = re.compile(r"线速度跟踪\s+([+-]?\d+(?:\.\d+)?).*?实际\s+vx\s+([+-]?\d+(?:\.\d+)?)\s*/\s*指令\s+vx\s+([+-]?\d+(?:\.\d+)?)")

    for line in text.splitlines():
        m = re_iter.search(line)
        if m:
            if cur and "ep_ret" in cur:
                iters.append(cur["iter"])
                ep_ret.append(cur["ep_ret"])
                survive.append(cur.get("survive", 0.0))
                falls.append(cur.get("falls", 0))
                track_lin.append(cur.get("track_lin", 0.0))
                vx.append(cur.get("vx", 0.0))
                cmd_vx.append(cur.get("cmd_vx", 0.0))
            cur = {"iter": int(m.group(1))}
            continue
        if not cur:
            continue
        m1 = re_ret.search(line)
        if m1:
            cur["ep_ret"] = float(m1.group(1))
            cur["survive"] = float(m1.group(2)) / max(float(m1.group(3)), 1.0)
            cur["falls"] = int(m1.group(4))
        m2 = re_vel.search(line)
        if m2:
            cur["track_lin"] = float(m2.group(1))
            cur["vx"] = float(m2.group(2))
            cur["cmd_vx"] = float(m2.group(3))

    if cur and "ep_ret" in cur:
        iters.append(cur["iter"])
        ep_ret.append(cur["ep_ret"])
        survive.append(cur.get("survive", 0.0))
        falls.append(cur.get("falls", 0))
        track_lin.append(cur.get("track_lin", 0.0))
        vx.append(cur.get("vx", 0.0))
        cmd_vx.append(cur.get("cmd_vx", 0.0))

    return {
        "file": filename,
        "iters": iters,
        "ep_return": ep_ret,
        "track_lin": track_lin,
        "vx": vx,
        "cmd_vx": cmd_vx,
        "falls": falls,
        "survive": survive,
    }


def quat_to_rot_and_euler(q: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """四元数 (w, x, y, z) 转旋转矩阵及 (roll, pitch, yaw) 弧度。"""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    roll = float(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = float(np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0)))
    yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return R, roll, pitch, yaw


class SimEngine:
    def __init__(self):
        self.start_time = time.time()
        self.dt = 0.002 * DECIMATION  # 0.02s (50Hz)
        self.speed_mult = 1.0
        self.running = True
        self.mode = "standby"  # standby | rl | relax | estop

        # 地形参数
        self.terrain_kind = "flat"  # flat | stairs
        self.step_h_cm = 10.0
        self.n_steps = 10
        self.tread_cm = 25.0

        self.model, self.stairs_meta = build_sim_model(0.0)
        self.data = mujoco.MjData(self.model)
        self.ghost_data = mujoco.MjData(self.model)
        self.base_id = self.model.body("base").id
        self.foot_body_ids = [self.model.body(f"{leg}_calf").id for leg in LEGS]

        # 几何资源缓存
        self.meshes_bin = pack_meshes_binary(self.model)
        self.robot_geoms = extract_robot_geoms(self.model)
        self.joint_ranges = [
            self.model.jnt_range[self.model.joint(name).id].tolist()
            for name in JOINT_NAMES
        ]

        # 控制与状态
        self.cmd = np.zeros(3, dtype=np.float64)  # vx, vy, wz
        self.speed_limit = 1.2
        self.q_target = DEFAULT_Q.copy()
        self.posture_from = DEFAULT_Q.copy()
        self.posture_to = DEFAULT_Q.copy()
        self.posture_t = 1.0
        self.posture_dur = 2.0
        self.jog_joint = -1

        self.payload = {"mass": 0.0, "x": 0.0, "y": 0.0, "z": 0.1}
        self.sustain_force = np.zeros(3, dtype=np.float64)
        self.sustain_left = 0.0

        self.falls = 0
        self._fall_cooldown = 0.0
        self._fall_reset_timer = -1.0

        # 策略加载
        self.checkpoints = scan_checkpoints()
        self.checkpoint_path = ""
        self.policy = None
        self.obs_taps = (0,)
        self.obs_hist = np.zeros((1, N_OBS), dtype=np.float32)
        self.last_action = np.zeros(12, dtype=np.float64)

        self.reset()
        if self.checkpoints:
            # 优先挑一个平地或楼梯常见模型作为默认预加载
            preferred = next(
                (p for p in self.checkpoints if "vel2" in p and p.endswith(".pt")),
                self.checkpoints[0],
            )
            self.load_checkpoint(preferred)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.34
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[7:] = DEFAULT_Q
        self.data.qvel[:] = 0.0
        self.q_target = DEFAULT_Q.copy()
        self.posture_from = DEFAULT_Q.copy()
        self.posture_to = DEFAULT_Q.copy()
        self.posture_t = 1.0
        self.last_action[:] = 0.0
        self.sustain_left = 0.0
        self._fall_reset_timer = -1.0
        self._apply_pd_gains()
        mujoco.mj_forward(self.model, self.data)
        self._reset_obs_hist()

    def _apply_pd_gains(self):
        if self.mode == "rl":
            kp, kd = KP, KD
        elif self.mode == "standby":
            kp, kd = 60.0, 2.0
        else:  # relax | estop
            kp, kd = 0.0, 0.0
        self.model.actuator_gainprm[:, 0] = kp
        self.model.actuator_biasprm[:, 1] = -kp
        self.model.actuator_biasprm[:, 2] = -kd

    def rebuild_terrain(self, kind: str, step_h_cm: float, n_steps: int, tread_cm: float):
        self.terrain_kind = kind
        self.step_h_cm = float(np.clip(step_h_cm, 2.0, 25.0))
        self.n_steps = int(np.clip(n_steps, 1, 30))
        self.tread_cm = float(np.clip(tread_cm, 15.0, 50.0))
        h = self.step_h_cm / 100.0 if kind == "stairs" else 0.0
        self.model, self.stairs_meta = build_sim_model(h, self.n_steps, self.tread_cm / 100.0)
        self.data = mujoco.MjData(self.model)
        self.ghost_data = mujoco.MjData(self.model)
        self.base_id = self.model.body("base").id
        self.foot_body_ids = [self.model.body(f"{leg}_calf").id for leg in LEGS]
        self.reset()

    def load_checkpoint(self, rel_path: str) -> dict:
        full = (ROOT / rel_path).resolve()
        if not full.exists():
            return {"ok": False, "error": f"不存在: {rel_path}"}
        try:
            pol, ck = load_policy(str(full))
            taps = tuple(ck.get("env_cfg", {}).get("obs_taps") or (0,))
            self.policy = pol
            self.obs_taps = taps
            self.checkpoint_path = rel_path
            self._reset_obs_hist()
            return {"ok": True, "checkpoint": rel_path, "obs_taps": list(taps)}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _compute_single_obs(self) -> np.ndarray:
        R, _, _, _ = quat_to_rot_and_euler(self.data.qpos[3:7])
        lin_w = self.data.qvel[:3]
        _ = R.T @ lin_w
        ang_b = self.data.qvel[3:6]
        g_b = -R[2, :]
        q = self.data.qpos[7:]
        dq = self.data.qvel[6:]
        scale = np.array([2.0, 2.0, 0.25], dtype=np.float64)
        obs = np.concatenate([
            ang_b * 0.25,
            g_b,
            self.cmd * scale,
            q - DEFAULT_Q,
            dq * 0.05,
            self.last_action,
        ]).astype(np.float32)
        return obs

    def _reset_obs_hist(self):
        hist_len = max(self.obs_taps) + 1
        obs0 = self._compute_single_obs()
        self.obs_hist = np.repeat(obs0[None, :], hist_len, axis=0)

    def _get_actor_obs(self) -> np.ndarray:
        if len(self.obs_taps) == 1:
            return self.obs_hist[0:1]
        return self.obs_hist[list(self.obs_taps), :].reshape(1, -1)

    def set_mode(self, mode: str):
        if mode not in ("standby", "rl", "relax", "estop"):
            return
        if mode == "rl" and self.policy is None:
            return
        prev = self.mode
        self.mode = mode
        if mode == "standby" and prev != "standby":
            q_now = self.data.qpos[7:].copy()
            self.posture_from = q_now
            self.posture_to = DEFAULT_Q.copy()
            self.posture_t = 0.0
            self.q_target = q_now
        elif mode == "rl":
            self.last_action[:] = 0.0
            self._reset_obs_hist()
        self._apply_pd_gains()

    def trigger_posture(self, name: str, dur: float):
        if name not in POSTURE_PRESETS:
            return
        self.mode = "standby"
        self._apply_pd_gains()
        self.posture_from = self.data.qpos[7:].copy()
        self.posture_to = POSTURE_PRESETS[name].copy()
        self.posture_dur = max(0.1, float(dur))
        self.posture_t = 0.0

    def push_impulse(self, direction: str, mag: float):
        mag = float(np.clip(mag, 0.0, 5.0))
        dx, dy = {
            "front": (mag, 0.0),
            "back": (-mag, 0.0),
            "left": (0.0, mag),
            "right": (0.0, -mag),
        }.get(direction, (0.0, 0.0))
        self.data.qvel[0] += dx
        self.data.qvel[1] += dy

    def push_sustain(self, direction: str, force_n: float, dur_s: float):
        f = float(np.clip(force_n, 0.0, 1500.0))
        dur = float(np.clip(dur_s, 0.02, 5.0))
        fx, fy = {
            "front": (f, 0.0),
            "back": (-f, 0.0),
            "left": (0.0, f),
            "right": (0.0, -f),
        }.get(direction, (0.0, 0.0))
        self.sustain_force[:] = [fx, fy, 0.0]
        self.sustain_left = dur

    @torch.no_grad()
    def step_once(self):
        if not self.running:
            return

        # 1. 计算目标关节角
        if self.mode == "rl" and self.policy is not None:
            obs = self._get_actor_obs()
            act = self.policy.actor(torch.as_tensor(obs)).numpy()[0]
            act = np.clip(act, -10.0, 10.0)
            self.last_action = act.astype(np.float64)
            self.q_target = DEFAULT_Q + ACTION_SCALE * self.last_action
        elif self.mode == "standby":
            if self.posture_t < 1.0:
                self.posture_t = min(1.0, self.posture_t + self.dt / self.posture_dur)
                alpha = 0.5 - 0.5 * np.cos(np.pi * self.posture_t)
                self.q_target = (1.0 - alpha) * self.posture_from + alpha * self.posture_to
        self.data.ctrl[:] = self.q_target

        # 2. 外力：负载重力与偏移力矩 + 持续推力
        self.data.xfrc_applied[:] = 0.0
        R, _, _, _ = quat_to_rot_and_euler(self.data.qpos[3:7])
        if self.payload["mass"] > 0.0:
            f_world = np.array([0.0, 0.0, -self.payload["mass"] * 9.81], dtype=np.float64)
            r_body = np.array([self.payload["x"], self.payload["y"], self.payload["z"]], dtype=np.float64)
            r_world = R @ r_body
            tau_world = np.cross(r_world, f_world)
            self.data.xfrc_applied[self.base_id, :3] += f_world
            self.data.xfrc_applied[self.base_id, 3:] += tau_world

        if self.sustain_left > 0.0:
            self.data.xfrc_applied[self.base_id, :3] += self.sustain_force
            self.sustain_left = max(0.0, self.sustain_left - self.dt)

        # 3. 推进物理 (10 个子步)
        for _ in range(DECIMATION):
            mujoco.mj_step(self.model, self.data)

        # 4. 异常保护 & 摔倒检测
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            self.reset()
            return

        if self.obs_hist.shape[0] > 1:
            self.obs_hist[1:] = self.obs_hist[:-1]
        self.obs_hist[0] = self._compute_single_obs()

        # 摔倒检测（非松电机/急停模式下）
        self._fall_cooldown = max(0.0, self._fall_cooldown - self.dt)
        R_now, _, _, _ = quat_to_rot_and_euler(self.data.qpos[3:7])
        cos_tilt = R_now[2, 2]
        ground_z = self._approx_ground_z(float(self.data.qpos[0]), float(self.data.qpos[1]))
        fallen = (self.data.qpos[2] - ground_z < 0.14) or (cos_tilt < 0.5)
        if fallen and self.mode in ("rl", "standby"):
            if self._fall_cooldown <= 0.0:
                self.falls += 1
                self._fall_cooldown = 2.0
                if self.mode == "rl":
                    self._fall_reset_timer = 1.2
        if self._fall_reset_timer > 0.0:
            self._fall_reset_timer -= self.dt
            if self._fall_reset_timer <= 0.0:
                self.reset()

    def _approx_ground_z(self, x: float, y: float) -> float:
        if self.terrain_kind != "stairs" or abs(y) > STAIR_HALF_WIDTH:
            return 0.0
        h = self.step_h_cm / 100.0
        tread = self.tread_cm / 100.0
        if x < STAIR_X0:
            return 0.0
        far_x = STAIR_X0 + (self.n_steps - 1) * tread + tread + STAIR_TOP_LEN
        if x > far_x:
            return 0.0
        lvl = int(np.clip(np.floor((x - STAIR_X0) / tread) + 1, 0, self.n_steps))
        return lvl * h

    def build_telemetry(self) -> dict:
        R, roll, pitch, yaw = quat_to_rot_and_euler(self.data.qpos[3:7])
        lin_b = R.T @ self.data.qvel[:3]
        ang_b = self.data.qvel[3:6]

        # 13 个刚体位姿 (body 1..13)
        body_poses = [
            [
                float(self.data.xpos[i, 0]),
                float(self.data.xpos[i, 1]),
                float(self.data.xpos[i, 2]),
                float(self.data.xquat[i, 0]),
                float(self.data.xquat[i, 1]),
                float(self.data.xquat[i, 2]),
                float(self.data.xquat[i, 3]),
            ]
            for i in range(1, self.model.nbody)
        ]

        # 参考蓝影姿态（保持当前机身 xy/yaw 与目标站姿）
        self.ghost_data.qpos[:] = self.data.qpos
        self.ghost_data.qpos[2] = max(float(self.data.qpos[2]), self._approx_ground_z(float(self.data.qpos[0]), float(self.data.qpos[1])) + 0.30)
        self.ghost_data.qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        self.ghost_data.qpos[7:] = DEFAULT_Q
        mujoco.mj_kinematics(self.model, self.ghost_data)
        ghost_poses = [
            [
                float(self.ghost_data.xpos[i, 0]),
                float(self.ghost_data.xpos[i, 1]),
                float(self.ghost_data.xpos[i, 2]),
                float(self.ghost_data.xquat[i, 0]),
                float(self.ghost_data.xquat[i, 1]),
                float(self.ghost_data.xquat[i, 2]),
                float(self.ghost_data.xquat[i, 3]),
            ]
            for i in range(1, self.model.nbody)
        ]

        # 关节锚点世界坐标 (用于 3D 标签投影)
        joint_anchors = [
            [float(self.data.xanchor[i, 0]), float(self.data.xanchor[i, 1]), float(self.data.xanchor[i, 2])]
            for i in range(1, self.model.njnt)
        ]

        # 足端坐标（approx：calf 末端的 [0, 0, -0.213]）
        foot_offset = np.array([0.0, 0.0, -0.213], dtype=np.float64)
        foot_pos = []
        for bid in self.foot_body_ids:
            Rm = self.data.xmat[bid].reshape(3, 3)
            fp = self.data.xpos[bid] + Rm @ foot_offset
            foot_pos.append([float(fp[0]), float(fp[1]), float(fp[2])])

        return {
            "t": round(float(self.data.time), 3),
            "uptime_h": round((time.time() - self.start_time) / 3600.0, 2),
            "running": self.running,
            "mode": self.mode,
            "falls": self.falls,
            "checkpoint": self.checkpoint_path,
            "terrain": {
                "kind": self.terrain_kind,
                "step_h_cm": self.step_h_cm,
                "n_steps": self.n_steps,
                "tread_cm": self.tread_cm,
            },
            "base_pos": [float(self.data.qpos[0]), float(self.data.qpos[1]), float(self.data.qpos[2])],
            "vx": float(lin_b[0]),
            "vy": float(lin_b[1]),
            "wz": float(ang_b[2]),
            "cmd": [float(self.cmd[0]), float(self.cmd[1]), float(self.cmd[2])],
            "roll": float(np.degrees(roll)),
            "pitch": float(np.degrees(pitch)),
            "yaw": float(np.degrees(yaw)),
            "q": [round(float(x), 4) for x in self.data.qpos[7:]],
            "q_tgt": [round(float(x), 4) for x in self.q_target],
            "dq": [round(float(x), 4) for x in self.data.qvel[6:]],
            "tau": [round(float(x), 3) for x in self.data.actuator_force],
            "body_poses": body_poses,
            "ghost_poses": ghost_poses,
            "joint_anchors": joint_anchors,
            "foot_pos": foot_pos,
        }


CLIENTS: set[ServerConnection] = set()


def make_http_response(status: int, content_type: str, body: bytes) -> Response:
    headers = Headers({
        "Content-Type": content_type,
        "Content-Length": str(len(body)),
        "Cache-Control": "no-cache",
    })
    reason = "OK" if status == 200 else "Not Found"
    return Response(status, reason, headers, body)


async def handle_ws(ws: ServerConnection, engine: SimEngine):
    CLIENTS.add(ws)
    try:
        # 连接时立刻同步一次楼梯描述
        await ws.send(json.dumps({"type": "stairs", "stairs": engine.stairs_meta}))
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            action = msg.get("action")
            if action == "cmd":
                lim = engine.speed_limit
                engine.cmd[0] = float(np.clip(msg.get("vx", 0.0), -lim, lim))
                engine.cmd[1] = float(np.clip(msg.get("vy", 0.0), -0.8, 0.8))
                engine.cmd[2] = float(np.clip(msg.get("wz", 0.0), -2.0, 2.0))
            elif action == "speed_limit":
                engine.speed_limit = float(np.clip(msg.get("value", 1.2), 0.2, 2.5))
            elif action == "mode":
                engine.set_mode(msg.get("mode", "standby"))
            elif action == "connect":
                engine.running = bool(msg.get("running", True))
            elif action == "reset":
                engine.reset()
            elif action == "estop":
                engine.set_mode("estop")
            elif action == "speed_mult":
                engine.speed_mult = float(np.clip(msg.get("value", 1.0), 0.25, 2.0))
            elif action == "posture":
                engine.trigger_posture(msg.get("name", "stand"), msg.get("dur", 2.0))
            elif action == "jog":
                idx = int(msg.get("index", -1))
                val = float(msg.get("value", 0.0))
                if 0 <= idx < 12:
                    if engine.mode != "standby":
                        engine.set_mode("standby")
                    engine.posture_t = 1.0
                    lo, hi = engine.joint_ranges[idx]
                    engine.q_target[idx] = float(np.clip(val, lo, hi))
            elif action == "terrain":
                engine.rebuild_terrain(
                    msg.get("kind", "flat"),
                    float(msg.get("step_h_cm", 10.0)),
                    int(msg.get("n_steps", 10)),
                    float(msg.get("tread_cm", 25.0)),
                )
                payload = json.dumps({"type": "stairs", "stairs": engine.stairs_meta})
                await asyncio.gather(*(c.send(payload) for c in list(CLIENTS)), return_exceptions=True)
            elif action == "checkpoint":
                res = engine.load_checkpoint(msg.get("path", ""))
                await ws.send(json.dumps({"type": "checkpoint_status", **res}))
            elif action == "payload":
                engine.payload = {
                    "mass": float(np.clip(msg.get("mass", 0.0), 0.0, 25.0)),
                    "x": float(np.clip(msg.get("x", 0.0), -0.4, 0.4)),
                    "y": float(np.clip(msg.get("y", 0.0), -0.3, 0.3)),
                    "z": float(np.clip(msg.get("z", 0.1), -0.2, 0.4)),
                }
            elif action == "push":
                engine.push_impulse(msg.get("dir", "front"), float(msg.get("mag", 1.0)))
            elif action == "sustain_push":
                engine.push_sustain(
                    msg.get("dir", "front"),
                    float(msg.get("force", 200.0)),
                    float(msg.get("dur", 0.15)),
                )
    finally:
        CLIENTS.discard(ws)


async def sim_broadcast_loop(engine: SimEngine):
    tick_dt = engine.dt
    next_tick = time.perf_counter()
    frame = 0
    while True:
        now = time.perf_counter()
        if now < next_tick:
            await asyncio.sleep(next_tick - now)
        next_tick = max(time.perf_counter(), next_tick + tick_dt / engine.speed_mult)

        engine.step_once()
        frame += 1
        # 以 ~25-33 Hz 推送遥测（每次控制步 50 Hz，隔帧或每帧推，单机本地极快）
        if CLIENTS and frame % 2 == 0:
            pkt = json.dumps({"type": "state", **engine.build_telemetry()})
            await asyncio.gather(*(c.send(pkt) for c in list(CLIENTS)), return_exceptions=True)


async def run_server(host: str, port: int):
    engine = SimEngine()

    def process_request(connection: ServerConnection, request):
        parsed = urlparse(request.path)
        path = parsed.path
        if path == "/ws":
            return None
        if path == "/":
            return make_http_response(200, "text/html; charset=utf-8", (WEB_DIR / "index.html").read_bytes())
        if path == "/style.css":
            return make_http_response(200, "text/css; charset=utf-8", (WEB_DIR / "style.css").read_bytes())
        if path == "/app.js":
            return make_http_response(200, "application/javascript; charset=utf-8", (WEB_DIR / "app.js").read_bytes())
        if path == "/api/meshes.bin":
            return make_http_response(200, "application/octet-stream", engine.meshes_bin)
        if path == "/api/init":
            meta = {
                "robot": "unitree_go2",
                "joint_names": JOINT_NAMES,
                "joint_ranges": engine.joint_ranges,
                "torque_limits": TORQUE_LIMIT.tolist(),
                "default_q": DEFAULT_Q.tolist(),
                "checkpoints": engine.checkpoints,
                "current_checkpoint": engine.checkpoint_path,
                "logs": scan_logs(),
                "geoms": engine.robot_geoms,
                "stairs": engine.stairs_meta,
            }
            return make_http_response(200, "application/json; charset=utf-8", json.dumps(meta).encode("utf-8"))
        if path == "/api/log":
            qs = parse_qs(parsed.query)
            fname = qs.get("file", [""])[0]
            data = parse_training_log(fname)
            return make_http_response(200, "application/json; charset=utf-8", json.dumps(data).encode("utf-8"))
        return make_http_response(404, "text/plain; charset=utf-8", b"Not Found")

    torch.set_num_threads(2)
    async with serve(lambda ws: handle_ws(ws, engine), host, port, process_request=process_request):
        print(f"[WebLab] 服务已启动: http://{host}:{port}/  (本机可访问 http://127.0.0.1:{port}/)", flush=True)
        await sim_broadcast_loop(engine)


def main():
    ap = argparse.ArgumentParser(description="rl01 Unitree Go2 WebLab 控制台")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8800)
    args = ap.parse_args()
    try:
        asyncio.run(run_server(args.host, args.port))
    except KeyboardInterrupt:
        print("\n[WebLab] 已停止", flush=True)


if __name__ == "__main__":
    main()
