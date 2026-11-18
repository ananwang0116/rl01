import sys, numpy as np, torch
from rl01.play import load_policy
from rl01.posture import measure
V4 = "runs/go2_vel2_v4/model_400.pt"; V5 = "runs/go2_vel2_v5/model_400.pt"
policies = {"v4(蹭地)": V4, "v5(抬腿)": V5, "混合 0.8~1.0": f"{V4},{V5},0.8,1.0"}
speeds = [float(x) for x in sys.argv[1:]] or [0.3, 0.5, 0.8, 0.9, 1.0, 1.5, 2.0]
print(f"{'策略':<13}{'指令':>5}{'实际vx':>8}{'摔倒':>5}{'触地%':>7}{'抬脚峰值cm':>11}{'摆动高cm':>9}{'外偏mm':>8}{'FL-FR小腿':>10}{'FL-FR外偏':>10}")
for name, path in policies.items():
    ac, ck = load_policy(path)
    for vx in speeds:
        out, s, _ = measure(ac, ck, vx, steps=400, skip=150, n=32)
        L = list(out.values())
        print(f"{name:<13}{vx:5.1f}{s['vx']/max(s['steps'],1):+8.2f}{s['falls']:5d}{100*np.mean([d['contact'] for d in L]):7.0f}"
              f"{np.mean([d['lift_cm'] for d in L]):11.1f}{np.nanmean([d['swing_z_cm'] for d in L]):9.1f}"
              f"{np.mean([d['outward_mm'] for d in L]):+8.1f}{out['FL']['calf']-out['FR']['calf']:+10.3f}{out['FL']['outward_mm']-out['FR']['outward_mm']:+10.1f}")
    print()
