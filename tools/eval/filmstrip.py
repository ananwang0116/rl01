"""连续帧：侧视一行 + 前视一行，时间间隔 --every 个控制步（默认 4 步 = 0.08 s），用来真实地看抬脚和晃动。
用法：MUJOCO_GL=egl uv run python tools/eval/filmstrip.py <检查点> --vx 0.5 --out out.png [--n 10] [--every 4]
可以同时给多个 --vx：每个速度一张图，文件名加 _vx。"""
import argparse
import mujoco, numpy as np, torch
import PIL.Image as I
from rl01.env import build_train_model
from rl01.play import load_policy, make_env, policy_action, set_cmd

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint"); ap.add_argument("--vx", type=float, nargs="+", default=[0.5]); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=10); ap.add_argument("--every", type=int, default=4); ap.add_argument("--warm", type=int, default=250)
a = ap.parse_args()
torch.set_num_threads(2)
ac, ck = load_policy(a.checkpoint)
model = build_train_model(visual=True); data = mujoco.MjData(model)
r = mujoco.Renderer(model, 300, 400)
for vx in a.vx:
    env = make_env(ck, 1, 0.0); set_cmd(env, vx)
    side, front = [], []
    for k in range(a.warm + a.n * a.every):
        env.step(policy_action(ac, env)); set_cmd(env, vx)
        if k >= a.warm and (k - a.warm) % a.every == 0:
            mujoco.mj_setState(model, data, env.state[0], mujoco.mjtState.mjSTATE_FULLPHYSICS); mujoco.mj_forward(model, data)
            yaw = np.degrees(np.arctan2(env.R[0, 1, 0], env.R[0, 0, 0]))
            for lst, az, dist in ((side, 90, 1.1), (front, 180, 1.3)):
                cam = mujoco.MjvCamera(); cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.15]
                cam.distance, cam.elevation, cam.azimuth = dist, -4, yaw + az
                r.update_scene(data, cam); lst.append(r.render().copy())
    img = np.concatenate([np.concatenate(side, 1), np.concatenate(front, 1)], 0)
    path = a.out if len(a.vx) == 1 else a.out.replace(".png", f"_{vx}.png")
    I.fromarray(img).save(path); print("已保存", path, f"上：侧视，下：前视；相邻帧 {a.every * 0.02:.2f}s，指令 vx={vx}")
