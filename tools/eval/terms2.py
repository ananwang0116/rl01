import numpy as np, torch
from rl01.play import load_policy, make_env, set_cmd
ac, ck = load_policy("runs/go2_vel2_v4/model_400.pt")
def run(vx, label):
    env = make_env(ck, 128, 0.6); env.cfg.only_positive_rewards = False; set_cmd(env, vx)
    acc = {}; n = 0
    for k in range(300):
        with torch.no_grad(): a = ac.actor(torch.as_tensor(env.get_obs()[0])).numpy()
        env.step(a); set_cmd(env, vx)
        if k >= 100:
            n += 1
            for t, v in env.term_means.items(): acc[t] = acc.get(t, 0) + v
    acc = {t: v / n for t, v in acc.items()}
    print(f"--- v4 策略 {label}  合计 {sum(acc.values()):+.4f}/步")
    print("  " + "  ".join(f"{t}={v:+.4f}" for t, v in acc.items() if abs(v) > 5e-5))
run(0.0, "站立"); run(1.0, "vx=1.0"); run(2.0, "vx=2.0")
# 摆动脚离地高度（统计量）
env = make_env(ck, 64, 0.0); set_cmd(env, 1.0)
for k in range(300):
    with torch.no_grad(): a = ac.actor(torch.as_tensor(env.get_obs()[0])).numpy()
    env.step(a); set_cmd(env, 1.0)
s = env.pop_stats(); print("摆动脚统计 (vx=1.0): 样本数 %d，平均离地 %.2f cm" % (s["swing_n"], 100 * s["swing_z_sum"] / max(s["swing_n"], 1)))
