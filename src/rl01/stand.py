"""Go2 站立测试：用 PD 控制把 12 个关节保持在 home 姿态。

用法：
    uv run python src/rl01/stand.py            # 无界面，跑 5 秒，打印结果
    uv run python src/rl01/stand.py --view     # 打开查看器，一直开着，关窗口才退出
"""
import argparse

import numpy as np

from rl01.sim import build_model, home_q, simulate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--seconds", type=float, default=5.0, help="无界面时的仿真时长")
    args = parser.parse_args()

    model, data = build_model()
    q_target = home_q(model)

    print(f"timestep={model.opt.timestep}s  nq={model.nq}  nu={model.nu}")
    print("关节顺序（qpos[7:] / ctrl 的顺序）:")
    for i in range(model.nu):
        jnt = model.actuator_trnid[i, 0]
        print(f"  {i:2d} {model.joint(jnt).name:14s} 目标 {q_target[i]:+.2f}")

    def log(d):
        print(f"t={d.time:4.1f}s  机身高度={d.qpos[2]:.3f} m")

    simulate(model, data, lambda t: q_target, seconds=args.seconds, view=args.view, on_second=log)

    if not args.view:
        height = data.qpos[2]
        err = np.abs(data.qpos[7:] - q_target).max()
        print(f"\n结束：机身高度={height:.3f} m  最大关节误差={err:.3f} rad")
        print("结果:", "站稳" if height > 0.2 and err < 0.3 else "摔倒或偏差过大")


if __name__ == "__main__":
    main()
