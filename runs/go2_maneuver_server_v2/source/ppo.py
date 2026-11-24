"""PPO（参照 rsl_rl 的做法）：非对称 actor-critic、GAE、自适应学习率（按 KL）、裁剪价值损失。"""
import torch
import torch.nn as nn


def mlp(n_in, hidden, n_out):
    layers, d = [], n_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ELU()]
        d = h
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, n_obs, n_critic_obs, n_act, hidden=(512, 256, 128), init_std=1.0):
        super().__init__()
        self.actor = mlp(n_obs, hidden, n_act)
        self.critic = mlp(n_critic_obs, hidden, 1)
        self.std = nn.Parameter(init_std * torch.ones(n_act))

    def dist(self, obs):
        mean = self.actor(obs)
        return torch.distributions.Normal(mean, self.std.clamp(min=1e-3).expand_as(mean))

    def value(self, cobs):
        return self.critic(cobs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, cobs):
        d = self.dist(obs)
        a = d.sample()
        return a, d.log_prob(a).sum(-1), self.value(cobs), d.mean, d.stddev


class PPO:
    def __init__(self, ac, lr=1e-3, gamma=0.99, lam=0.95, clip=0.2, epochs=5, minibatches=4,
                 entropy_coef=0.01, value_coef=1.0, max_grad_norm=1.0, desired_kl=0.01, mirror=None, sym_coef=0.0):
        self.ac, self.lr = ac, lr
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.minibatches = epochs, minibatches
        self.entropy_coef, self.value_coef = entropy_coef, value_coef
        self.max_grad_norm, self.desired_kl = max_grad_norm, desired_kl
        self.mirror_obs, self.mirror_act = mirror if mirror else (None, None)
        self.sym_coef = sym_coef
        self.opt = torch.optim.Adam(ac.parameters(), lr=lr)

    def update(self, st, last_value):
        """st: 字典，张量形状 [T, N, ...]：obs cobs act logp val rew done mu sigma。"""
        T, N = st["rew"].shape
        adv = torch.zeros_like(st["rew"])
        last = torch.zeros(N, device=adv.device)
        for t in reversed(range(T)):
            nxt_val = last_value if t == T - 1 else st["val"][t + 1]
            nonterm = 1.0 - st["done"][t]
            delta = st["rew"][t] + self.gamma * nxt_val * nonterm - st["val"][t]
            last = delta + self.gamma * self.lam * nonterm * last
            adv[t] = last
        ret = adv + st["val"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        flat = lambda x: x.reshape(T * N, *x.shape[2:])
        obs, cobs, act, logp0 = flat(st["obs"]), flat(st["cobs"]), flat(st["act"]), flat(st["logp"])
        val0, mu0, sig0, adv, ret = flat(st["val"]), flat(st["mu"]), flat(st["sigma"]), flat(adv), flat(ret)

        B = T * N
        mb = B // self.minibatches
        s_loss = v_loss = kl_mean = sym_mean = 0.0
        n = 0
        for _ in range(self.epochs):
            perm = torch.randperm(B, device=obs.device)
            for k in range(self.minibatches):
                i = perm[k * mb : (k + 1) * mb]
                d = self.ac.dist(obs[i])
                logp = d.log_prob(act[i]).sum(-1)
                value = self.ac.value(cobs[i])
                with torch.no_grad():  # 自适应学习率：KL 太大就降，太小就升
                    kl = torch.sum(torch.log(d.stddev / sig0[i] + 1e-5)
                                   + (sig0[i] ** 2 + (mu0[i] - d.mean) ** 2) / (2 * d.stddev ** 2) - 0.5, -1).mean()
                    if self.desired_kl:
                        if kl > self.desired_kl * 2:
                            self.lr = max(1e-5, self.lr / 1.5)
                        elif 0 < kl < self.desired_kl / 2:
                            self.lr = min(1e-2, self.lr * 1.5)
                        for g in self.opt.param_groups:
                            g["lr"] = self.lr
                ratio = torch.exp(logp - logp0[i])
                surr = -torch.min(adv[i] * ratio, adv[i] * ratio.clamp(1 - self.clip, 1 + self.clip)).mean()
                v_clip = val0[i] + (value - val0[i]).clamp(-self.clip, self.clip)
                vl = torch.max((value - ret[i]) ** 2, (v_clip - ret[i]) ** 2).mean()
                loss = surr + self.value_coef * vl - self.entropy_coef * d.entropy().sum(-1).mean()
                if self.mirror_obs is not None:  # 镜像对称损失：actor(镜像观测) 应当等于 镜像(actor(观测))
                    target = self.mirror_act(d.mean).detach()
                    sym = ((self.ac.actor(self.mirror_obs(obs[i])) - target) ** 2).mean()
                    loss = loss + self.sym_coef * sym
                    sym_mean += sym.item()
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.opt.step()
                s_loss += surr.item(); v_loss += vl.item(); kl_mean += kl.item(); n += 1
        return dict(surrogate=s_loss / n, value=v_loss / n, kl=kl_mean / n, lr=self.lr, sym=sym_mean / n)
