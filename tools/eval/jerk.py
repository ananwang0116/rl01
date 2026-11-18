import numpy as np, torch
from rl01.play import load_policy, make_env, set_cmd, policy_action
ac, ck = load_policy("runs/go2_vel2_v5/model_400.pt")
print("v5 第 400 步，64 环境 × 600 步（12 秒）；抖动 = 相邻两步动作的最大关节变化 |Δa|；角速度单位 rad/s")
print(f"{'指令':>5}{'推力':>6}{'|Δa|中位':>9}{'|Δa|99.9%':>10}{'|Δa|>3 占比':>12}{'俯仰角速度99.9%':>16}{'横滚角速度99.9%':>16}{'大抖动里推力后0.2s内占比':>24}")
for vx in (1.5, 2.0):
    for push in (0.0, 0.6):
        env = make_env(ck, 64, push); set_cmd(env, vx)
        prev = None; since = np.full(64, 10**6); jerks = []; wy = []; wx = []; after = []
        for k in range(600):
            a = policy_action(ac, env); env.step(a); set_cmd(env, vx)
            pushed = np.linalg.norm(env.push_delta, axis=1) > 0
            since = np.where(pushed, 0, since + 1)
            if prev is not None and k >= 100:
                j = np.abs(a - prev).max(1); jerks.append(j); after.append(since <= 10)
                wy.append(np.abs(env.ang_b[:, 1])); wx.append(np.abs(env.ang_b[:, 0]))
            prev = a
        J = np.concatenate(jerks); A = np.concatenate(after); big = J > 3.0
        frac_after = (A[big].mean() if big.any() else float("nan"))
        print(f"{vx:5.1f}{push:6.1f}{np.median(J):9.2f}{np.percentile(J,99.9):10.2f}{big.mean():12.4f}{np.percentile(np.concatenate(wy),99.9):16.2f}{np.percentile(np.concatenate(wx),99.9):16.2f}{frac_after:24.2f}")
