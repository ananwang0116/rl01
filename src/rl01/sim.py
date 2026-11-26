"""Go2 的 MuJoCo 公共部分：加载模型（可加楼梯）、PD 控制、查看器循环。

关节顺序（qpos[7:] / ctrl）：FL, FR, RL, RR，每条腿 hip, thigh, calf。
"""
import time
from pathlib import Path

import mujoco
import numpy as np

SCENE = Path(__file__).resolve().parents[2] / "models" / "unitree_go2" / "scene.xml"
D1_MAX_SCENE = SCENE.parent.parent / "agibot_d1_max" / "scene.xml"
LEGS = ["FL", "FR", "RL", "RR"]
KP, KD = 60.0, 2.0  # 静态站立需要较大刚度；kp=20 时机身会塌到 0.195 m


def build_model(step_h=0.0, n_steps=4, tread=0.30, x0=0.6, half_width=0.6, top_len=1.0, *, scene=SCENE):
    """加载 MuJoCo 场景（默认 Go2）；step_h>0 时在前方（+x）加楼梯。

    每一级是一个从地面堆到该级高度的 box；最后一级向后延伸 top_len 当平台。
    注意 MuJoCo 的 box size 是半长。
    """
    spec = mujoco.MjSpec.from_file(str(scene))
    if step_h > 0:
        for i in range(n_steps):
            depth = tread + (top_len if i == n_steps - 1 else 0.0)
            height = (i + 1) * step_h
            spec.worldbody.add_geom(
                name=f"stair{i}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[depth / 2, half_width, height / 2],
                pos=[x0 + i * tread + depth / 2, 0.0, height / 2],
                rgba=[0.75, 0.6, 0.4, 1.0],
            )
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.keyframe("home").id)
    mujoco.mj_forward(model, data)
    return model, data


def home_q(model):
    return model.key("home").qpos[7:].copy()


def pd_step(model, data, q_target, kp=KP, kd=KD):
    """一个物理步：motor 执行器的 ctrl 就是力矩，超过 ctrlrange 会被截断。"""
    q = data.qpos[7:]
    dq = data.qvel[6:]
    data.ctrl[:] = kp * (q_target - q) - kd * dq
    mujoco.mj_step(model, data)


def simulate(model, data, target_fn, seconds=None, view=False, on_second=None, kp=KP, kd=KD):
    """使用力矩 PD 控制。target_fn(t) 返回各关节的目标角。"""
    def step():
        pd_step(model, data, target_fn(data.time), kp, kd)

    run_simulation(model, data, step, seconds=seconds, view=view, on_second=on_second)


def run_simulation(model, data, step, *, seconds=None, view=False, on_second=None):
    """重复调用 step() 推进仿真。

    view=True：打开查看器并**一直开着**，直到你关掉窗口（seconds 被忽略），按真实时间播放。
    view=False：无界面，跑 seconds 秒，尽快算完。
    on_second(data)：每个仿真秒调用一次（打印用）。
    """
    dt = model.opt.timestep
    last_sec = -1

    def tick():
        nonlocal last_sec
        step()
        sec = int(data.time)
        if on_second and sec != last_sec:
            last_sec = sec
            on_second(data)

    if not view:
        for _ in range(int(seconds / dt)):
            tick()
        return

    import mujoco.viewer

    sync_every = max(1, int(round(1 / 60 / dt)))  # 约 60 帧/秒
    with mujoco.viewer.launch_passive(model, data) as viewer:
        wall0 = time.perf_counter()
        sim0 = data.time
        while viewer.is_running():
            for _ in range(sync_every):
                tick()
            viewer.sync()
            ahead = (data.time - sim0) - (time.perf_counter() - wall0)  # 仿真比真实时间快了多少
            if ahead > 0:
                time.sleep(ahead)


def tilt(model, data):
    """机身倾角（rad）：机身 z 轴与世界 z 轴的夹角。"""
    bid = model.body("base").id
    return float(np.arccos(np.clip(data.xmat[bid][8], -1, 1)))
