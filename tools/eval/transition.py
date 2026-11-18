import numpy as np, torch
from rl01.play import load_policy, make_env, set_cmd, policy_action
from rl01.sim import LEGS
V4 = "runs/go2_vel2_v4/model_400.pt"; V5 = "runs/go2_vel2_v5/model_400.pt"
policies = {"v4": V4, "v5": V5, "混合": f"{V4},{V5},0.8,1.0"}
# 指令随时间的变化 (秒 -> vx)
scen = {
    "阶跃 0.6→1.4": lambda t: 0.6 if t < 4 else 1.4,
    "阶跃 1.4→0.6": lambda t: 1.4 if t < 4 else 0.6,
    "斜坡 0.5→1.5（10 秒）": lambda t: 0.5 + min(t / 10.0, 1.0) * 1.0,
    "停在 0.9（混合中点）": lambda t: 0.9,
    "来回抖动 0.7↔1.1（每 1 秒）": lambda t: 0.7 if int(t) % 2 == 0 else 1.1,
}
print(f"带 0.6 m/s 随机推力，64 环境，每个场景 12 秒")
print(f"{'场景':<22}{'策略':<5}{'摔倒':>5}{'最大倾角°':>10}{'动作突变max':>12}{'动作突变均值':>13}{'稳态误差':>9}")
for sname, cmd_fn in scen.items():
    for pname, path in policies.items():
        ac, ck = load_policy(path); env = make_env(ck, 64, 0.6)
        prev = None; jerk = []; tilt = []; err = []
        for k in range(600):
            t = k * env.dt; vx = cmd_fn(t); set_cmd(env, vx)
            a = policy_action(ac, env); env.step(a)
            if prev is not None: jerk.append(np.abs(a - prev).max(1))   # 相邻两步动作变化量（每个环境取最大关节）
            prev = a
            tilt.append(np.degrees(np.arccos(np.clip(env.R[:, 2, 2], -1, 1))).max())
            if t > 2: err.append(abs(env.lin_b[:, 0].mean() - vx))
        s = env.pop_stats(); jerk = np.concatenate(jerk)
        print(f"{sname:<22}{pname:<5}{s['falls']:5d}{max(tilt):10.1f}{jerk.max():12.2f}{jerk.mean():13.3f}{np.mean(err):9.3f}")
    print()
