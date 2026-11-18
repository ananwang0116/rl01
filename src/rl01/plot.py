"""画训练曲线。训练中或训练后都能用；竖线标出保存检查点的位置（10%、20%、...）。

用法：
    uv run python src/rl01/plot.py runs/<名字>            # 保存 runs/<名字>/curves.png
    uv run python src/rl01/plot.py runs/<名字> --show     # 同时弹出窗口
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

import numpy as np


def load(run_dir):
    with open(run_dir / "metrics.csv") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def smooth(y, k=20):
    if len(y) < k:
        return y
    ok = ~np.isnan(y)
    y2 = np.where(ok, y, 0.0)
    num = np.convolve(y2, np.ones(k), "same")
    den = np.convolve(ok.astype(float), np.ones(k), "same")
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(args.run_dir)
    d = load(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    total = cfg["args"]["iterations"]
    ckpts = sorted(int(p.stem.split("_")[1]) for p in run_dir.glob("model_*.pt"))
    x = d["iteration"]

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    panels = [
        ("每步平均奖励", ["reward_step"]),
        ("线速度/角速度跟踪（越接近 1 越好）", ["track_lin", "track_ang"]),
        ("实际 vx 与指令 vx（m/s）", ["vx", "cmd_vx"]),
        ("指令上限课程 vmax（m/s）", ["vmax"]),
        ("回合存活步数（满 = 1000）", ["ep_len"]),
        ("摔倒次数 / 迭代", ["falls"]),
        ("价值损失", ["value"]),
        ("KL 与学习率（自适应）", ["kl", "lr"]),
        ("动作标准差（探索程度）", ["std"]),
        ("足端横向外偏 mm（正 = 外八，0 最好）", ["foot_out_mm", "foot_abs_mm"]),
        ("摆动脚离地高度 cm（越高越不蹭地，真实约 6-8）", ["swing_z_cm"]),
        ("左右镜像对称损失（越小越对称）", ["sym"]),
    ]
    if cfg["env"].get("stairs"):
        panels[3] = ("楼梯课程阶段 (0-8)", ["stage"])
        panels[4] = (f"回合存活步数（超时 = {int(cfg['env']['episode_s'] / 0.02)}）", ["ep_len"])
        panels.append(("每迭代成功 / 结束回合", ["stair_success", "stair_done"]))
    panels = [(t, [k for k in keys if k in d]) for t, keys in panels]  # 旧运行没有新列：跳过
    panels = [(t, keys) for t, keys in panels if keys]
    rows = (len(panels) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(16, 3.4 * rows), constrained_layout=True)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    for ax, (title, keys) in zip(axes.ravel(), panels):
        for i, k in enumerate(keys):
            ax.plot(x, d[k], alpha=0.25, lw=0.8, color=f"C{i}")  # 浅色=原始，深色=平滑（20 迭代）
            ax.plot(x, smooth(d[k]), lw=1.6, label=k, color=f"C{i}")
        for c in ckpts:
            ax.axvline(c, color="gray", ls=":", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, total)
        if len(keys) > 1:
            ax.legend(fontsize=8)
        if keys in (["value"], ["kl", "lr"], ["sym"]):
            ax.set_yscale("log")
    fig.suptitle(f"{run_dir.name}   已完成 {int(x[-1])}/{total} 迭代，检查点 {ckpts}（虚线）", fontsize=12)
    out = run_dir / "curves.png"
    fig.savefig(out, dpi=110)
    print("已保存", out)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
