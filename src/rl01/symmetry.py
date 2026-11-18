"""Go2 左右镜像：把"状态/动作"沿机身 y 轴翻转（左 <-> 右），用来给 PPO 加对称损失。

关节顺序 FL, FR, RL, RR，每条腿 hip, thigh, calf：
  - 腿互换：FL <-> FR，RL <-> RR
  - hip（外展，绕 x 轴）取反；thigh、calf（绕 y 轴）不变
  - 默认姿态的 hip 是 +0.1 / -0.1（左右反号），所以 (q - DEFAULT_Q) 的 hip 同样只需取反
actor 观测（45）布局：[0:3] 角速度 [3:6] 投影重力 [6:9] 指令(vx, vy, yaw) [9:21] q-默认 [21:33] dq [33:45] 上一动作
  - 角速度 (wx, wy, wz) -> (-wx, wy, -wz)；投影重力 (gx, gy, gz) -> (gx, -gy, gz)；指令 (vx, vy, yaw) -> (vx, -vy, -yaw)
"""
import numpy as np
import torch

LEG_PERM = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])  # 镜像后第 i 个关节取自原来的第 LEG_PERM[i] 个
JOINT_SIGN = np.array([-1, 1, 1] * 4, dtype=np.float32)  # hip 取反


def mirror_joints(x):
    """x 的最后一维是 12 个关节量（角度、角速度、动作）。numpy 或 torch 都可以。"""
    if isinstance(x, torch.Tensor):
        return x[..., LEG_PERM] * torch.as_tensor(JOINT_SIGN, device=x.device, dtype=x.dtype)
    return x[..., LEG_PERM] * JOINT_SIGN


def mirror_obs(obs):
    """actor 观测的镜像（45 维）。"""
    sign3 = lambda a, b, c: torch.as_tensor([a, b, c], device=obs.device, dtype=obs.dtype)
    return torch.cat([
        obs[..., 0:3] * sign3(-1, 1, -1),
        obs[..., 3:6] * sign3(1, -1, 1),
        obs[..., 6:9] * sign3(1, -1, -1),
        mirror_joints(obs[..., 9:21]),
        mirror_joints(obs[..., 21:33]),
        mirror_joints(obs[..., 33:45]),
    ], dim=-1)


def mirror_action(a):
    return mirror_joints(a)


def mirror_state(s):
    """rollout 状态向量 (N,38)：time, qpos(x y z, quat w x y z, 12 关节), qvel(世界系线速度, 机身系角速度, 12 关节速度)。"""
    m = s.copy()
    m[:, 2] = -s[:, 2]  # y
    m[:, 5] = -s[:, 5]  # 四元数 x
    m[:, 7] = -s[:, 7]  # 四元数 z
    m[:, 8:20] = mirror_joints(s[:, 8:20])
    m[:, 21] = -s[:, 21]  # 世界系线速度 vy
    m[:, 23] = -s[:, 23]  # 机身系角速度 wx
    m[:, 25] = -s[:, 25]  # 机身系角速度 wz
    m[:, 26:38] = mirror_joints(s[:, 26:38])
    return m
