"""Go2 和 D1 Max 的站立测试。

用法：
    uv run python src/rl01/stand.py            # 无界面，跑 5 秒，打印结果
    uv run python src/rl01/stand.py --view     # 打开查看器，一直开着，关窗口才退出
    uv run python src/rl01/stand.py --robot d1-max --view
"""
import argparse

import mujoco
import numpy as np

from rl01.sim import D1_MAX_SCENE, SCENE, build_model, pd_step, run_simulation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=("go2", "d1-max"), default="go2")
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--seconds", type=float, default=5.0, help="无界面时的仿真时长")
    args = parser.parse_args()

    d1_max = args.robot == "d1-max"
    model, data = build_model(scene=D1_MAX_SCENE if d1_max else SCENE)
    home = model.key("home")
    joint_ids = model.actuator_trnid[:, 0]
    qpos_ids = model.jnt_qposadr[joint_ids]
    dof_ids = model.jnt_dofadr[joint_ids]
    wheel = np.array([d1_max and model.joint(j).name.endswith("_FOOT_JOINT") for j in joint_ids])
    q_target = home.qpos[qpos_ids]
    control = home.ctrl.copy()

    print(f"{args.robot}  timestep={model.opt.timestep}s  nq={model.nq}  nu={model.nu}")
    print("执行器顺序（ctrl 索引）")
    for i, joint_id in enumerate(joint_ids):
        target = f"{control[i]:+.2f} rad/s" if wheel[i] else f"{q_target[i]:+.2f} rad"
        print(f"  {i:2d} {model.joint(joint_id).name:18s} 目标 {target}")

    def step():
        if d1_max:
            data.ctrl[:] = control
            mujoco.mj_step(model, data)
        else:
            pd_step(model, data, q_target)

    def log(d):
        print(f"t={d.time:4.1f}s  机身高度={d.qpos[2]:.3f} m")

    run_simulation(model, data, step, seconds=args.seconds, view=args.view, on_second=log)

    if not args.view:
        height = data.qpos[2]
        err = np.abs(data.qpos[qpos_ids[~wheel]] - q_target[~wheel]).max()
        upright = data.xmat[model.jnt_bodyid[0], 8] > np.cos(np.deg2rad(20))
        print(f"\n结束：机身高度={height:.3f} m  最大关节误差={err:.3f} rad")
        stable = height > 0.2 and err < 0.3
        if d1_max:
            wheel_speed = np.abs(data.qvel[dof_ids[wheel]]).max()
            print(f"最大轮速={wheel_speed:.3f} rad/s")
            stable = stable and upright and height > 0.4 and wheel_speed < 0.5
        print("结果:", "站稳" if stable else "摔倒或偏差过大")


if __name__ == "__main__":
    main()
