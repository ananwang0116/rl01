"""逐条腿测量姿态（用来判断"哪条腿不对劲、外八多少"，改前改后用同一个标准对比）。

输出（每条腿，稳态平均）：
  hip/thigh/calf 相对默认姿态的偏差（rad）；髋外展 hip 偏差越大越"外八"
  足端横向外偏 mm：足端在机身系下的 y，相对大腿关节 y（±0.142 m）向外多出多少，正 = 向外撇
  触地占比、抬脚最高点（cm）、关节角标准差（摆动幅度）
再给出左右对称差（FL-FR、RL-RR）。

用法：
    uv run python src/rl01/posture.py runs/<名字>/model_<n>.pt                 # 速度 0 / 1.0 / 2.0，无推力
    uv run python src/rl01/posture.py runs/<名字>/model_<n>.pt --png out.png   # 再渲染一张前视+俯视图（需要 MUJOCO_GL=egl）
"""
import argparse

import numpy as np
import torch

from rl01.env import DEFAULT_Q
from rl01.play import load_policy, make_env, policy_action, set_cmd
from rl01.sim import LEGS

THIGH_Y = 0.0465 + 0.0955  # 大腿关节在机身系下的横向位置（左 +，右 −）


def measure(ac, ck, vx, steps=400, skip=150, n=32, push=0.0):
    env = make_env(ck, n, push)
    set_cmd(env, vx)
    q, foot_b, contact, foot_z, foot_w_all = [], [], [], [], []
    g_all, w_all, h_all = [], [], []  # 晃动：机身系重力方向、角速度、机身高度（都要 copy，env 里是会被覆盖的缓冲）
    for k in range(steps):
        env.step(policy_action(ac, env))
        set_cmd(env, vx)
        if k >= skip:
            q.append(env.q.copy())
            pos = env.state[:, 1:4]
            fw = env.sens_buf[:, -1, :12].reshape(n, 4, 3).copy()  # 必须 copy：reshape 只是视图，sens_buf 每步被覆盖，不 copy 则列表里所有时刻都是最后一步
            foot_b.append(np.einsum("nji,nkj->nki", env.R, fw - pos[:, None, :]))
            foot_z.append(fw[:, :, 2])
            g_all.append(env.g_b.copy()); w_all.append(env.ang_b.copy()); h_all.append(env.state[:, 3].copy())
            foot_w_all.append(fw)
    q, foot_b, foot_z = np.array(q), np.array(foot_b), np.array(foot_z)
    fw_all = np.array(foot_w_all)  # (T, n, 4, 3) 世界系足端位置
    speed = np.linalg.norm(np.diff(fw_all[:, :, :, :2], axis=0), axis=-1) / env.dt  # (T-1, n, 4) 足端水平速度 m/s
    z_mid = foot_z[1:]
    s = env.pop_stats()
    g, w, h = np.array(g_all), np.array(w_all), np.array(h_all)  # (T, n, 3) (T, n, 3) (T, n)
    rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
    s["wobble"] = dict(
        roll_deg=rms(np.degrees(np.arctan2(g[..., 1], -g[..., 2]))), pitch_deg=rms(np.degrees(np.arctan2(-g[..., 0], -g[..., 2]))),
        roll_rate=rms(w[..., 0]), pitch_rate=rms(w[..., 1]), height_std_mm=float(np.mean(np.std(h, axis=0)) * 1000),
    )
    sign = np.array([1, -1, 1, -1])  # FL FR RL RR：左 +y，右 −y
    out = {}
    for i, leg in enumerate(LEGS):
        dq = q[:, :, 3 * i : 3 * i + 3] - DEFAULT_Q[3 * i : 3 * i + 3]
        out[leg] = dict(
            hip=dq[:, :, 0].mean(), thigh=dq[:, :, 1].mean(), calf=dq[:, :, 2].mean(),
            hip_std=q[:, :, 3 * i].std(), thigh_std=q[:, :, 3 * i + 1].std(), calf_std=q[:, :, 3 * i + 2].std(),
            outward_mm=(sign[i] * foot_b[:, :, i, 1] - THIGH_Y).mean() * 1000,
            contact=(foot_z[:, :, i] < 0.03).mean(), lift_cm=np.percentile(foot_z[:, :, i], 95) * 100,
            swing_frac=(speed[:, :, i] > 0.5).mean(),  # 脚在快速移动（摆动）的时间占比
            swing_z_cm=(z_mid[:, :, i][speed[:, :, i] > 0.5].mean() * 100) if (speed[:, :, i] > 0.5).any() else float('nan'),  # 快速移动的脚的离地高度
        )
    return out, s, env


def report(vx, out, s):
    print(f"\n=== 指令 vx={vx:.1f} m/s   实际 vx={s['vx'] / max(s['steps'], 1):+.2f}  摔倒 {s['falls']}")
    print(f"{'腿':>3} {'hip偏差':>8} {'thigh偏差':>9} {'calf偏差':>9} {'外偏mm':>7} {'触地%':>6} {'抬脚cm':>7} {'摆动%':>7} {'摆动高cm':>9} {'hip摆幅':>8} {'thigh摆幅':>9} {'calf摆幅':>9}")
    for leg, d in out.items():
        print(f"{leg:>3} {d['hip']:+8.3f} {d['thigh']:+9.3f} {d['calf']:+9.3f} {d['outward_mm']:+7.1f} {100 * d['contact']:6.1f} "
              f"{d['lift_cm']:7.1f} {100 * d['swing_frac']:7.1f} {d['swing_z_cm']:9.1f} {d['hip_std']:8.3f} {d['thigh_std']:9.3f} {d['calf_std']:9.3f}")
    for a, b in (("FL", "FR"), ("RL", "RR")):
        print(f"  左右差 {a}-{b}: thigh {out[a]['thigh'] - out[b]['thigh']:+.3f}  calf {out[a]['calf'] - out[b]['calf']:+.3f}  "
              f"外偏 {out[a]['outward_mm'] - out[b]['outward_mm']:+.1f} mm  触地 {100 * (out[a]['contact'] - out[b]['contact']):+.1f}%  "
              f"抬脚 {out[a]['lift_cm'] - out[b]['lift_cm']:+.1f} cm")
    print(f"  蹭地指标：摆动脚平均离地 {np.nanmean([d['swing_z_cm'] for d in out.values()]):.1f} cm（脚中心静止时约 1.1 cm，真实小跑摆动高度约 6～8 cm），摆动占比 {100 * np.mean([d['swing_frac'] for d in out.values()]):.0f}%")
    wb = s["wobble"]
    print(f"  晃动指标（越小越稳）：横滚 RMS {wb['roll_deg']:.2f}°，俯仰 RMS {wb['pitch_deg']:.2f}°，横滚角速度 {wb['roll_rate']:.3f}，俯仰角速度 {wb['pitch_rate']:.3f} rad/s，机身高度起伏 {wb['height_std_mm']:.1f} mm")
    print(f"  整体：平均外偏 {np.mean([d['outward_mm'] for d in out.values()]):+.1f} mm，最大 |hip偏差| {max(abs(d['hip']) for d in out.values()):.3f} rad")


def render(ck, vx, path):
    import mujoco
    import PIL.Image as I

    from rl01.env import build_train_model

    ac, _ = load_policy(ck["path"])
    env = make_env(ck, 1, 0.0)
    set_cmd(env, vx)
    for _ in range(250):
        env.step(policy_action(ac, env))
        set_cmd(env, vx)
    model = build_train_model(visual=True)
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, env.state[0], mujoco.mjtState.mjSTATE_FULLPHYSICS)
    mujoco.mj_forward(model, data)
    yaw = np.degrees(np.arctan2(env.R[0, 1, 0], env.R[0, 0, 0]))  # 机身朝向，镜头跟着转
    imgs = []
    for az_off, el, dist in ((180, -8, 1.3), (180 + 90, -8, 1.5), (180, -89, 1.6)):  # 前视、侧视、俯视
        cam = mujoco.MjvCamera()
        cam.lookat[:] = data.qpos[:3]
        cam.distance, cam.elevation, cam.azimuth = dist, el, yaw + az_off
        r = mujoco.Renderer(model, 360, 480)
        r.update_scene(data, cam)
        imgs.append(r.render())
    I.fromarray(np.concatenate(imgs, axis=1)).save(path)
    print("已保存", path, "（左：从正前方看；中：侧视；右：俯视）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--vx", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    ap.add_argument("--png", default=None)
    args = ap.parse_args()
    torch.set_num_threads(2)
    ac, ck = load_policy(args.checkpoint)
    ck["path"] = args.checkpoint
    print(f"检查点 {args.checkpoint}（迭代 {ck['iteration']}），无推力，确定性动作，32 个环境")
    for vx in args.vx:
        out, s, _ = measure(ac, ck, vx)
        report(vx, out, s)
    if args.png:
        render(ck, args.vx[1] if len(args.vx) > 1 else args.vx[0], args.png)


if __name__ == "__main__":
    main()
