"""Go2 开环小跑（trot）步态测试：不用任何反馈，只验证接触、摩擦、楼梯场景是否合理。

trot = 对角步态：FL+RR 一组、FR+RL 一组，两组相位差半个周期。
足端在大腿-小腿平面里走轨迹（x 向前，z 向上，原点在大腿关节），用解析逆运动学转成关节角。

用法：
    uv run python src/rl01/walk.py                       # 平地，无界面，跑 8 秒并打印
    uv run python src/rl01/walk.py --stairs 0.05         # 前方 5 cm 台阶
    uv run python src/rl01/walk.py --stairs 0.05 --view  # 打开查看器，一直开着
"""
import argparse

import mujoco
import numpy as np

from rl01.sim import LEGS, build_model, home_q, simulate, tilt

L = 0.213  # 大腿、小腿长度
PHASE = {"FL": 0.0, "RR": 0.0, "FR": 0.5, "RL": 0.5}  # 对角两条腿同相


def leg_ik(x, z):
    """足端 (x, z)（相对大腿关节）-> (thigh, calf)。公式用 home 姿态校验过：(0, -0.2648) -> (0.9, -1.8)。"""
    d2 = x * x + z * z
    cos_knee = np.clip((2 * L * L - d2) / (2 * L * L), -1, 1)
    bend = np.pi - np.arccos(cos_knee)  # 小腿相对大腿弯了多少（calf = -bend）
    phi = np.arctan2(-x, -z)  # 足端相对竖直向下的偏角
    return phi + bend / 2, -bend


def foot_traj(p, stride, z0, lift, duty):
    """一个周期内的足端轨迹：支撑相向后直线滑动，摆动相抬起向前。"""
    if p < duty:
        s = p / duty
        return stride / 2 - stride * s, z0
    s = (p - duty) / (1 - duty)
    return -stride / 2 + stride * (1 - np.cos(np.pi * s)) / 2, z0 + lift * np.sin(np.pi * s)


def make_target(q_home, period, stride, z0, lift, duty, ramp=1.0):
    def target(t):
        q = np.zeros(12)
        for i, leg in enumerate(LEGS):
            p = (t / period + PHASE[leg]) % 1.0
            x, z = foot_traj(p, stride, z0, lift, duty)
            th, ca = leg_ik(x, z)
            q[3 * i : 3 * i + 3] = [0.0, th, ca]
        a = min(t / ramp, 1.0)  # 前 1 秒从 home 姿态平滑过渡到步态，避免一开始就猛抬腿
        return (1 - a) * q_home + a * q

    return target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stairs", type=float, default=0.0, help="台阶高度（米），0 表示平地")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--stride", type=float, default=0.10, help="步幅（米）")
    ap.add_argument("--period", type=float, default=0.5, help="步态周期（秒）")
    ap.add_argument("--lift", type=float, default=None, help="抬脚高度（米），默认 max(0.06, 台阶+0.04)")
    ap.add_argument("--duty", type=float, default=0.6, help="支撑相占比")
    ap.add_argument("--kp", type=float, default=200.0, help="开环步态需要高刚度：kp=80 时机身塌到 0.215 m、几乎走不动")
    ap.add_argument("--kd", type=float, default=3.0)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--view", action="store_true")
    args = ap.parse_args()

    lift = args.lift if args.lift is not None else max(0.06, args.stairs + 0.04)
    model, data = build_model(step_h=args.stairs, n_steps=args.steps)
    target = make_target(home_q(model), args.period, args.stride, -0.24, lift, args.duty)

    v = args.stride / (args.duty * args.period)
    print(f"trot：周期 {args.period}s 步幅 {args.stride}m 抬脚 {lift}m 占空比 {args.duty} -> 理论速度约 {v:.2f} m/s")
    if args.stairs:
        print(f"楼梯：{args.steps} 级，每级高 {args.stairs*100:.0f} cm，踏面 0.30 m，起点 x=0.6 m")

    x_start = data.qpos[0]
    state = {"max_tilt": 0.0}

    def log(d):
        state["max_tilt"] = max(state["max_tilt"], tilt(model, d))
        print(f"t={d.time:4.1f}s  x={d.qpos[0]:+.2f}  y={d.qpos[1]:+.2f}  高度={d.qpos[2]:.3f}  倾角={np.degrees(tilt(model, d)):4.1f}°")

    simulate(model, data, target, seconds=args.seconds, view=args.view, on_second=log, kp=args.kp, kd=args.kd)

    if not args.view:
        fell = data.qpos[2] < 0.15 or tilt(model, data) > 0.8
        print(f"\n前进 {data.qpos[0] - x_start:+.2f} m，侧偏 {data.qpos[1]:+.2f} m，最终高度 {data.qpos[2]:.3f} m，"
              f"最大倾角 {np.degrees(state['max_tilt']):.1f}°")
        print("结果:", "摔倒" if fell else "没摔倒")


if __name__ == "__main__":
    main()
