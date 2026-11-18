"""v5 在新奖励（低速步态：目标高度随速度 + 上界 + 加重横滚俯仰惩罚）下各项每步贡献，用来定权重量级。
用法：uv run python tools/eval/terms3.py [检查点] """
import sys, torch
from rl01.play import load_policy, make_env, set_cmd
torch.set_num_threads(2)
path = sys.argv[1] if len(sys.argv) > 1 else "runs/go2_vel2_v5/model_400.pt"
ac, ck = load_policy(path)
def run(vx):
    env = make_env(ck, 128, 0.0)
    c = env.cfg; c.only_positive_rewards = False; c.speed_clear = True
    c.w_foot_clear = -150.0; c.w_foot_cap = -300.0; c.w_orient_low = -20.0; c.w_ang_vel_low = -0.4
    set_cmd(env, vx); acc = {}; n = 0
    for k in range(300):
        with torch.no_grad(): a = ac.actor(torch.as_tensor(env.get_obs()[0])).numpy()
        env.step(a); set_cmd(env, vx)
        if k >= 100:
            n += 1
            for t, v in env.term_means.items(): acc[t] = acc.get(t, 0) + v
    acc = {t: v / n for t, v in acc.items()}
    print(f"--- vx={vx}  合计 {sum(acc.values()):+.4f}/步")
    print("  " + "  ".join(f"{t}={v:+.4f}" for t, v in acc.items() if abs(v) > 5e-5))
for vx in (0.0, 0.3, 0.5, 0.8, 1.0, 2.0): run(vx)
