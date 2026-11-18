"""按速度指令混合两个策略的动作：低速用 low，高速用 high，中间线性过渡。

两个策略必须是同一种网络、同一套观测和动作约定（例如 v4 与从 v4 微调出来的 v5）。
混合权重只由指令的前进速度决定：
    w = clip((vx - v_lo) / (v_hi - v_lo), 0, 1)        # w = 1 表示完全用 high
    action = (1 - w) * low(obs) + w * high(obs)
观测里的指令项是 vx * 2（见 env.py），所以这里要除回去。两个策略都收到同一个观测，
其中"上一个动作"是实际执行的（混合后的）动作。
"""
import torch


class BlendedPolicy:
    def __init__(self, low, high, v_lo=0.8, v_hi=1.0):
        assert v_hi > v_lo
        self.low, self.high, self.v_lo, self.v_hi = low, high, v_lo, v_hi

    def weight(self, obs):
        vx = obs[..., 6] / 2.0
        return ((vx - self.v_lo) / (self.v_hi - self.v_lo)).clamp(0.0, 1.0)

    def actor(self, obs):
        w = self.weight(obs).unsqueeze(-1)
        return (1.0 - w) * self.low.actor(obs) + w * self.high.actor(obs)

    def eval(self):
        return self


def save_blend(low_path, high_path, out_path, v_lo=0.8, v_hi=1.0):
    """把两个检查点打包成一个文件。"model" 字段放的是 high 的权重（旧的加载代码拿到的是高速策略）。"""
    low = torch.load(low_path, map_location="cpu")
    high = torch.load(high_path, map_location="cpu")
    torch.save(dict(
        blend=dict(v_lo=v_lo, v_hi=v_hi, low_model=low["model"], high_model=high["model"],
                   low_src=str(low_path), high_src=str(high_path)),
        model=high["model"], iteration=high["iteration"], vmax=high["vmax"], env_cfg=high["env_cfg"],
    ), out_path)


if __name__ == "__main__":
    import sys
    # uv run python src/rl01/blend.py <低速.pt> <高速.pt> <输出.pt> [v_lo v_hi]
    args = sys.argv[1:]
    assert len(args) in (3, 5), "用法：blend.py <低速.pt> <高速.pt> <输出.pt> [v_lo v_hi]"
    save_blend(args[0], args[1], args[2], *(float(x) for x in args[3:]))
    print("已保存", args[2])
