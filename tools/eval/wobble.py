import numpy as np, torch
from rl01.play import load_policy, make_env, set_cmd, policy_action
V4 = "runs/go2_vel2_v4/model_400.pt"; V5 = "runs/go2_vel2_v5/model_400.pt"
policies = {"v4": V4, "v5": V5, "混合": f"{V4},{V5},0.8,1.0"}
print("无推力，64 环境，400 步（丢掉前 150 步）；全部是时间序列的标准差/均方根，越小越稳")
print(f"{'策略':<5}{'指令':>5}{'实际vx':>8}{'横滚RMS°':>10}{'俯仰RMS°':>10}{'横滚角速度':>11}{'俯仰角速度':>11}{'侧向速度RMS':>12}{'机身高度std mm':>15}{'动作突变均值':>13}")
for speed in (0.3, 0.5, 0.8, 1.0, 1.5, 2.0):
    for pname, path in policies.items():
        ac, ck = load_policy(path); env = make_env(ck, 64, 0.0); set_cmd(env, speed)
        roll=[]; pitch=[]; wx=[]; wy=[]; vy=[]; h=[]; jerk=[]; vxs=[]; prev=None
        for k in range(400):
            a = policy_action(ac, env); env.step(a); set_cmd(env, speed)
            if prev is not None and k >= 150: jerk.append(np.abs(a - prev).mean(1))
            prev = a
            if k >= 150:
                g = env.g_b                                  # 机身系重力方向
                roll.append(np.degrees(np.arctan2(g[:, 1], -g[:, 2])))
                pitch.append(np.degrees(np.arctan2(-g[:, 0], -g[:, 2])))
                wx.append(env.ang_b[:, 0]); wy.append(env.ang_b[:, 1]); vy.append(env.lin_b[:, 1]); h.append(env.state[:, 3]); vxs.append(env.lin_b[:, 0].mean())
        rms = lambda x: float(np.sqrt(np.mean(np.square(np.array(x)))))
        hs = float(np.mean(np.std(np.array(h), axis=0)) * 1000)   # 每个环境机身高度随时间的标准差，再对环境取平均
        print(f"{pname:<5}{speed:5.1f}{np.mean(vxs):+8.2f}{rms(roll):10.2f}{rms(pitch):10.2f}{rms(wx):11.3f}{rms(wy):11.3f}{rms(vy):12.3f}{hs:15.1f}{np.mean(np.concatenate(jerk)):13.3f}")
    print()
