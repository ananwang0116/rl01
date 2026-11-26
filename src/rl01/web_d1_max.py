"""D1 Max live WebLab sessions, using the trained MuJoCo environment."""
import asyncio
import csv
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch
from websockets.exceptions import ConnectionClosed

from rl01.d1_max_env import JOINT_NAMES, N_ACT, STATE
from rl01.d1_max_omni import OmniEnv, build_model
from rl01.train_d1_max_omni import load_policy

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = Path("runs/d1_free_vt_r7_s3/model_1050.pt")
DIRECTIONS = {"front": (1., 0.), "back": (-1., 0.), "left": (0., 1.), "right": (0., -1.)}


def scan_d1_checkpoints(root=ROOT) -> list[str]:
    runs_dir = root / "runs"
    if not runs_dir.exists():
        return []
    pts = sorted(runs_dir.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p.relative_to(root)) for p in pts
            if p.relative_to(runs_dir).parts[0].startswith(("d1_", "d1max_", "d1_max_"))]


class D1Preview:
    def __init__(self, checkpoint, pack_meshes, extract_geoms, stairs_geometry):
        self.root = ROOT
        self.checkpoint = checkpoint
        self.pack_meshes = pack_meshes
        self.extract_geoms = extract_geoms
        self.stairs_geometry = stairs_geometry
        self.model = None

    def resolve_checkpoint(self):
        if self.checkpoint.is_file():
            return self.checkpoint
        pts = scan_d1_checkpoints(self.root)
        if pts:
            self.checkpoint = (self.root / pts[0]).resolve()
            return self.checkpoint
        return None

    def load(self):
        if self.model is None:
            ckpt = self.resolve_checkpoint()
            if ckpt is None:
                raise FileNotFoundError("No D1 Max checkpoint found under runs/")
            actor, cfg = load_policy(ckpt)
            model = build_model(visual=True)
            meshes = self.pack_meshes(model)
            self.actor, self.cfg, self.meshes = actor, cfg, meshes
            self.geoms = self.extract_geoms(model)
            self.model = model

    def select_checkpoint(self, rel_path: str) -> dict:
        try:
            target = (self.root / rel_path).resolve()
            runs_root = (self.root / "runs").resolve()
            if runs_root not in target.parents or target.suffix != ".pt" or not target.is_file():
                return {"ok": False, "error": f"Invalid checkpoint: {rel_path}"}
            actor, cfg = load_policy(target)
            self.checkpoint = target
            self.actor = actor
            self.cfg = cfg
            return {
                "ok": True,
                "checkpoint": rel_path,
                "limits": [cfg.max_speed, cfg.max_lateral, cfg.max_yaw],
                "reverse_limit": cfg.max_reverse or cfg.max_speed,
                "wheel_speed_limit": cfg.wheel_speed_limit,
                "champ_gait": cfg.champ_gait,
                "logs": ["metrics.csv"] if target.with_name("metrics.csv").is_file() else [],
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def metadata(self):
        self.load()
        joints = [self.model.joint(name).id for name in JOINT_NAMES]
        actuators = [self.model.actuator(name).id for name in JOINT_NAMES]
        pts = scan_d1_checkpoints(self.root)
        try:
            cur = str(self.checkpoint.resolve().relative_to(self.root))
        except ValueError:
            cur = self.checkpoint.name
        if cur not in pts:
            pts = [cur, *pts]
        return {
            "robot": "d1_max", "joint_names": JOINT_NAMES, "geoms": self.geoms, "stairs": [],
            "joint_ranges": [self.model.jnt_range[j].tolist() if self.model.jnt_limited[j] else None
                             for j in joints],
            "torque_limits": self.model.actuator_forcerange[actuators, 1].tolist(),
            "target_types": ["velocity" if name.endswith("_FOOT_JOINT") else "position" for name in JOINT_NAMES],
            "checkpoints": pts, "current_checkpoint": cur,
            "logs": ["metrics.csv"] if self.checkpoint.with_name("metrics.csv").is_file() else [],
            "limits": [self.cfg.max_speed, self.cfg.max_lateral, self.cfg.max_yaw],
            "reverse_limit": self.cfg.max_reverse or self.cfg.max_speed,
            "wheel_speed_limit": self.cfg.wheel_speed_limit,
            "champ_gait": self.cfg.champ_gait,
        }

    def training_metrics(self):
        path = self.checkpoint.with_name("metrics.csv")
        if not path.is_file():
            return None
        with path.open(newline="") as source:
            rows = list(csv.DictReader(source))
        columns = ("iteration", "reward", "vx_mae", "vy_mae", "yaw_mae", "lift_mm", "tilt_deg", "falls")
        return {column: [float(row[column]) for row in rows] for column in columns}

    async def handle(self, ws):
        self.load()
        session = D1Session(self)
        receiver = asyncio.create_task(session.receive(ws))
        try:
            await ws.send(json.dumps(session.terrain_packet()))
            next_tick = time.monotonic()
            frame = 0
            while not receiver.done():
                now = time.monotonic()
                if now < next_tick:
                    await asyncio.sleep(next_tick - now)
                next_tick = max(time.monotonic(), next_tick + session.env.dt / session.speed_mult)
                session.step()
                frame += 1
                if frame % 2 == 0:
                    await ws.send(json.dumps(session.telemetry(), allow_nan=False))
            # Propagate unexpected receive errors to the WebSocket handler.
            await receiver
        except ConnectionClosed:
            pass
        finally:
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)
            session.env.close()


class D1Session:
    def __init__(self, preview):
        self.preview = preview
        self.actor = preview.actor
        self.cfg = preview.cfg
        self.stairs_geometry = preview.stairs_geometry
        self.visual = preview.model
        self.data = mujoco.MjData(self.visual)
        self.ghost = mujoco.MjData(self.visual)
        self.env = OmniEnv(1, preview.cfg, training=False)
        self.limits = np.array([preview.cfg.max_speed, preview.cfg.max_lateral, preview.cfg.max_yaw])
        self.lower_limits = -self.limits.copy()
        if preview.cfg.max_reverse is not None:
            self.lower_limits[0] = -preview.cfg.max_reverse
        self.wheel_geoms = [self.visual.geom(f"{leg}_FOOT_LINK_collision").id
                            for leg in ("FBL", "FAR", "RBL", "RAR")]
        self.joints = [self.visual.joint(name).id for name in JOINT_NAMES]
        self.actuators = [self.visual.actuator(name).id for name in JOINT_NAMES]
        self.q_ids = self.visual.jnt_qposadr[self.joints]
        self.dq_ids = self.visual.jnt_dofadr[self.joints]
        self.limit_mask = self.visual.actuator_ctrllimited.astype(bool)
        self.wheel_actuators = [self.visual.actuator(name).id for name in JOINT_NAMES if name.endswith("_FOOT_JOINT")]
        self.stairs = []
        self.terrain = dict(kind="flat", step_h_cm=10., n_steps=10, tread_cm=25.)
        self.payload = np.zeros(4)
        self.started = time.monotonic()
        self.running = True
        self.mode = "rl"
        self.speed_mult = 1.0
        self.reset()

    def reset(self):
        self.env.reset(np.array([0]))
        self.env.command.fill(0)
        self.command = np.zeros(3)
        self.hold_deadline = 0.0
        self.tap_direction = 0.0
        self.tap_steps = 0
        self.tap_target_yaw = 0.0
        self.falls = 0
        self.pending_reset = False
        self.peak_lift = 0.0
        self.sustain_force = np.zeros(3)
        self.sustain_left = 0.
        self.env.ctrl[0] = self.env.home_ctrl
        self.posture_target = self.env.home_ctrl.copy()
        self.posture_start = self.posture_target.copy()
        self.posture_elapsed = 0.
        self.posture_duration = 0.

    def terrain_packet(self):
        return {"type": "stairs", "stairs": self.stairs, "terrain": self.terrain}

    def rebuild_terrain(self, message):
        h, count, tread = [float(message[k]) for k in ("step_h_cm", "n_steps", "tread_cm")]
        if not np.isfinite([h, count, tread]).all():
            return None
        kind = message["kind"]
        if kind not in ("flat", "stairs"):
            return None
        h, count, tread = float(np.clip(h, 2, 25)), int(np.clip(count, 1, 30)), float(np.clip(tread, 15, 50))
        boxes = self.stairs_geometry(h / 100 if kind == "stairs" else 0., count, tread / 100)
        env = OmniEnv(1, self.cfg, training=False, world_boxes=boxes)
        self.env.close()
        self.env = env
        self.stairs = boxes
        self.terrain = dict(kind=kind, step_h_cm=h, n_steps=count, tread_cm=tread)
        self.reset()
        self.set_mode(self.mode)
        return self.terrain_packet()

    def set_mode(self, mode):
        self.mode = mode
        self.command.fill(0)
        self.tap_steps = 0
        self.env.tap_remaining.fill(0)
        self.env.tap_heading_error.fill(0)
        self.env.previous_yaw.fill(0)
        self.env.reset_gait(np.array([0]))
        # The rollout uses a session-local model copy.
        model = self.env.models[0]
        enabled = mode not in ("relax", "estop")
        model.actuator_gainprm[:, 0] = self.env.model.actuator_gainprm[:, 0] * enabled
        model.actuator_biasprm[:, :3] = self.env.model.actuator_biasprm[:, :3] * enabled

    def set_posture(self, name, duration):
        poses = {"crouch": (1.05, 2.03), "lie": (1.34, 2.63)}
        target = self.env.home_ctrl.copy()
        if name in poses:
            hip, knee = poses[name]
            for i, side in enumerate((1, 1, -1, -1)):
                target[4 * i + 1:4 * i + 3] = [side * hip, -side * knee]
        elif name == "zero":
            target.fill(0)
        elif name not in ("stand", "home"):
            return
        if not np.isfinite(duration):
            return
        self.set_mode("standby")
        self.posture_start = self.env.state[0, 1 + self.q_ids].copy()
        self.posture_start[self.wheel_actuators] = 0
        self.posture_target = target
        self.posture_elapsed = 0.
        self.posture_duration = float(np.clip(duration, .1, 5.))

    def jog(self, index, value):
        if not 0 <= index < len(self.actuators) or not np.isfinite(value):
            return
        if self.mode != "standby":
            self.set_mode("standby")
        self.posture_target = self.env.ctrl[0].copy()
        wheel_limit = self.cfg.wheel_speed_limit
        limits = [-wheel_limit, wheel_limit] if index in self.wheel_actuators else self.visual.actuator_ctrlrange[index]
        self.posture_target[index] = np.clip(value, *limits)
        self.posture_duration = 0.

    def ground_height(self, x, y):
        height = 0.
        for box in self.stairs:
            pos, size = box["pos"], box["size"]
            if abs(x - pos[0]) <= size[0] and abs(y - pos[1]) <= size[1]:
                height = max(height, pos[2] + size[2])
        return height

    def message(self, msg):
        if not isinstance(msg, dict):
            return
        action = msg.get("action")
        if action == "reset":
            self.reset()
        elif action == "checkpoint":
            res = self.preview.select_checkpoint(msg.get("path", ""))
            if res.get("ok"):
                self.actor = self.preview.actor
                self.cfg = self.preview.cfg
                self.limits = np.array([self.cfg.max_speed, self.cfg.max_lateral, self.cfg.max_yaw])
                self.lower_limits = -self.limits.copy()
                if self.cfg.max_reverse is not None:
                    self.lower_limits[0] = -self.cfg.max_reverse
                new_env = OmniEnv(1, self.cfg, training=False, world_boxes=self.stairs)
                self.env.close()
                self.env = new_env
                self.reset()
                self.set_mode(self.mode)
            return {"type": "checkpoint_status", **res}
        elif action == "terrain":
            return self.rebuild_terrain(msg)
        elif action in ("stop", "estop"):
            self.command.fill(0)
            self.tap_steps = 0
            self.env.tap_remaining.fill(0)
            self.env.tap_heading_error.fill(0)
            self.env.reset_gait(np.array([0]))
            if action == "estop":
                self.set_mode("estop")
                self.running = False
        elif action == "mode" and msg.get("mode") in ("standby", "rl", "relax"):
            if msg["mode"] == "standby":
                self.set_posture("stand", 1.)
            else:
                self.set_mode(msg["mode"])
        elif action == "posture":
            self.set_posture(msg["name"], float(msg["dur"]))
        elif action == "jog":
            self.jog(int(msg["index"]), float(msg["value"]))
        elif action == "connect":
            self.running = bool(msg.get("running", True))
            if not self.running:
                self.command.fill(0)
                self.tap_steps = 0
            elif self.mode == "estop":
                self.set_mode("rl")
        elif action == "speed_mult":
            value = float(msg["value"])
            if np.isfinite(value):
                self.speed_mult = float(np.clip(value, .25, 2))
        elif action in ("command", "cmd"):
            value = msg.get("value", [msg.get("vx", 0), msg.get("vy", 0), msg.get("wz", 0)])
            cmd = np.asarray(value, dtype=float)
            if cmd.shape == (3,) and np.isfinite(cmd).all():
                self.command[:] = np.clip(cmd, self.lower_limits, self.limits)
                self.hold_deadline = time.monotonic() + 0.35
        elif action == "tap" and self.mode == "rl" and self.running and self.tap_steps == 0 and self.env.tap_remaining[0] <= 0:
            sign = msg.get("direction")
            angle_deg = float(msg.get("angle_deg", 2.0))
            if sign in (-1, 1) and np.isfinite(angle_deg) and 0.5 <= angle_deg <= 45.0:
                self.tap_direction = sign
                self.tap_target_yaw = (self.env.measured_yaw()[0] + sign * np.radians(angle_deg) + np.pi) % (2 * np.pi) - np.pi
                self.tap_steps = 40
                self.hold_deadline = 0.0
                self.peak_lift = 0.0
        elif action == "payload":
            values = np.asarray([msg.get(k, 0) for k in ("mass", "x", "y", "z")], dtype=float)
            if np.isfinite(values).all():
                self.payload[:] = np.clip(values, [0, -.4, -.3, -.2], [25, .4, .3, .4])
        elif action == "push" and self.running:
            if "angle" in msg:
                angle = float(msg["angle"])
                if np.isfinite(angle):
                    self.env.push([0], azimuth=np.deg2rad(angle % 360))
            elif msg.get("dir") in DIRECTIONS:
                magnitude = float(msg["mag"])
                if np.isfinite(magnitude):
                    start = self.env.root_dof
                    self.env.state[0, start:start + 2] += np.clip(magnitude, 0, 5) * np.asarray(DIRECTIONS[msg["dir"]])
        elif action == "sustain_push" and self.running and msg.get("dir") in DIRECTIONS:
            force, duration = float(msg["force"]), float(msg["dur"])
            if np.isfinite([force, duration]).all():
                self.sustain_force[:2] = np.clip(force, 0, 1500) * np.asarray(DIRECTIONS[msg["dir"]])
                self.sustain_left = float(np.clip(duration, .02, 5))

    async def receive(self, ws):
        async for raw in ws:
            try:
                packet = self.message(json.loads(raw))
                if packet is not None:
                    await ws.send(json.dumps(packet))
            except (ValueError, TypeError, KeyError, OverflowError):
                continue

    @torch.inference_mode()
    def step(self):
        if not self.running:
            self.env.command.fill(0)
            return
        if self.pending_reset:
            self.env.reset(np.array([0]))
            self.command.fill(0)
            self.tap_steps = 0
            self.pending_reset = False
        cmd = self.env.command[0]
        if self.mode != "rl":
            cmd.fill(0)
        elif self.tap_steps:
            err = (self.tap_target_yaw - self.env.measured_yaw()[0] + np.pi) % (2 * np.pi) - np.pi
            if abs(err) <= np.radians(0.15):
                self.tap_steps = 0
                cmd.fill(0)
            else:
                cmd[:] = (0, 0, np.clip(4.0 * err, -self.limits[2], self.limits[2]))
                self.tap_steps -= 1
        elif time.monotonic() < self.hold_deadline:
            cmd[:] = self.command
        else:
            cmd.fill(0)
        obs, _ = self.env.observations()
        action = self.actor(torch.from_numpy(obs)).numpy() if self.mode == "rl" else np.zeros((1, N_ACT))
        target = None
        if self.mode != "rl":
            target = self.posture_target
            if self.posture_elapsed < self.posture_duration:
                self.posture_elapsed = min(self.posture_elapsed + self.env.dt, self.posture_duration)
                mix = .5 - .5 * np.cos(np.pi * self.posture_elapsed / self.posture_duration)
                target = (1 - mix) * self.posture_start + mix * target
        _, _, info = self.env.step(action, control_targets=target, external_wrench=self.external_wrench())
        self.pending_reset = self.mode == "rl" and bool(info["fall"][0])
        self.falls += int(self.pending_reset)
        if self.env.tap_remaining[0] > 0 or self.tap_steps:
            centers = self.env.sensors[0, -1, self.env.wheel_sensor:].reshape(4, 7)
            floor = np.array([self.ground_height(x, y) for x, y in centers[:, :2]])
            self.peak_lift = max(self.peak_lift, float((info["clearance"][0] - floor).max() * 1000))

    def external_wrench(self):
        mass = self.payload[0]
        if mass == 0 and self.sustain_left <= 0:
            return None
        model = self.env.model
        wrench = np.zeros((1, model.nbody, 6))
        force = wrench[0, self.env.base, :3]
        if mass:
            force[2] = -9.81 * mass
            arm = np.empty(3)
            q = self.env.state[0, self.env.root + 3:self.env.root + 7]
            mujoco.mju_rotVecQuat(arm, self.payload[1:] - model.body_ipos[self.env.base], q)
            wrench[0, self.env.base, 3:] = np.cross(arm, force)
        if self.sustain_left > 0:
            fraction = min(self.sustain_left / self.env.dt, 1.)
            force += fraction * self.sustain_force
            self.sustain_left -= self.env.dt
        return wrench

    def telemetry(self):
        env, data = self.env, self.data
        mujoco.mj_setState(self.visual, data, env.state[0], STATE)
        data.ctrl[:] = env.ctrl[0]
        limited = self.visual.actuator_ctrllimited.astype(bool)
        data.ctrl[limited] = np.clip(data.ctrl[limited], *self.visual.actuator_ctrlrange[limited].T)
        mujoco.mj_forward(self.visual, data)
        v, ang, _ = env.body_state()
        w, x, y, z = data.qpos[3:7]
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x*x + y*y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1., 1.))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y*y + z*z))
        ghost = self.ghost
        ghost.qpos[:] = self.visual.key("home").qpos
        ghost.qpos[:2] = data.qpos[:2]
        ghost.qpos[2] = max(data.qpos[2], ghost.qpos[2] + self.ground_height(*data.qpos[:2]))
        ghost.qpos[3:7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
        mujoco.mj_kinematics(self.visual, ghost)
        torque = np.zeros(len(self.actuators)) if self.mode in ("relax", "estop") else data.actuator_force[self.actuators]
        return {
            "type": "state", "running": self.running, "mode": self.mode,
            "uptime_h": round((time.monotonic() - self.started) / 3600, 2),
            "t": round(float(data.time), 2), "base_pos": data.qpos[:3].tolist(),
            "body_poses": np.column_stack((data.xpos[1:], data.xquat[1:])).tolist(),
            "ghost_poses": np.column_stack((ghost.xpos[1:], ghost.xquat[1:])).tolist(),
            "foot_pos": data.geom_xpos[self.wheel_geoms].tolist(),
            "joint_anchors": data.xanchor[self.joints].tolist(),
            "q": data.qpos[self.q_ids].tolist(), "dq": data.qvel[self.dq_ids].tolist(),
            "q_tgt": data.ctrl[self.actuators].tolist(), "tau": torque.tolist(),
            "cmd": env.command[0].tolist(), "vx": float(v[0, 0]), "vy": float(v[0, 1]), "wz": float(ang[0, 2]),
            "roll": float(np.degrees(roll)), "pitch": float(np.degrees(pitch)), "yaw": float(np.degrees(yaw)),
            "tap_peak_mm": self.peak_lift,
            "falls": self.falls, "tap_active": bool(self.tap_steps > 0 or env.tap_remaining[0] > 0),
        }
