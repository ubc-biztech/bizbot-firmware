"""TD3 with many parallel robots (FastTD3-style), with an optional privileged critic.

    .venv/bin/python -u model/mj/td3.py --out model/mj/runs/td3 --json '{"budget_s": 300}'

The actor sees only the ESP32's observation; with asym the twin critics also
see the true state and robot parameters. Each robot explores with its own
noise level, drawn between noise_min and noise_max at every reset, so some
robots act almost deterministically. Checkpoints load with ppo.load_policy.
"""

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(HERE))
from ppo import ACT, RunningNorm, make_env_cfg, mlp  # noqa: E402


@dataclass
class TD3Config:
    env: str = "mj"
    workers: int = 12
    per_worker: int = 256
    buffer: int = 2_000_000
    batch: int = 16384
    updates_per_step: int = 1
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.05
    policy_delay: int = 2
    nstep: int = 3
    noise_min: float = 0.01
    noise_max: float = 0.2
    target_noise: float = 0.1
    target_clip: float = 0.2
    actor_hidden: tuple = (128, 128)
    critic_hidden: tuple = (512, 256)
    activation: str = "elu"
    asym: bool = True
    critic_layernorm: bool = False   # LayerNorm after each critic hidden layer
    actor_priv: object = False       # teacher: True = the actor also sees the true state and parameters,
                                     # "params" = only the robot's parameters (needs asym)
    n_critics: int = 2               # >2: REDQ-style ensemble (targets: min of a random 2; actor: mean of all)
    learning_starts: int = 200_000
    init_actor: str = ""             # start the actor from this TD3 actor checkpoint (e.g. a distilled student)
    init_learner: str = ""           # start critic, target critic, critic optimiser and normalisers from a .learner.pt
    actor_freeze_updates: int = 0    # no actor updates until this many critic updates (let a reused critic settle)
    scale: float = 1.0
    curriculum_s: float = 0.0
    budget_s: float = 300.0
    max_steps: int = 10 ** 10
    ckpt_every: int = 5_000_000
    threads: int = 2
    device: str = "mps"
    seed: int = 0
    env_overrides: dict = field(default_factory=dict)
    rand_overrides: dict = field(default_factory=dict)  # Randomization to replace: {name: [lo, hi, nominal(, log)] or {"p": prob}}


PRIV_STATE = 10   # leading true-state entries of BalanceEnv._priv; the rest are the robot's parameters


class TanhActor(nn.Module):
    def __init__(self, obs, hidden, act):
        super().__init__()
        self.net = mlp(obs, hidden, 1, act)

    def forward(self, x):
        return torch.tanh(self.net(x))


def mlp_ln(i, hidden, o, act):
    layers, d = [], i
    for h in hidden:
        layers += [nn.Linear(d, h), nn.LayerNorm(h), act()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class Twin(nn.Module):
    def __init__(self, inp, hidden, act, layernorm=False):
        super().__init__()
        f = mlp_ln if layernorm else mlp
        self.q1, self.q2 = f(inp + 1, hidden, 1, act), f(inp + 1, hidden, 1, act)

    def forward(self, x, a):
        xa = torch.cat([x, a], -1)
        return self.q1(xa).squeeze(-1), self.q2(xa).squeeze(-1)


class Ensemble(nn.Module):
    """n Q-networks; forward returns (n, batch), or only the listed members."""

    def __init__(self, inp, hidden, act, layernorm, n):
        super().__init__()
        f = mlp_ln if layernorm else mlp
        self.qs = nn.ModuleList([f(inp + 1, hidden, 1, act) for _ in range(n)])

    def forward(self, x, a, only=None):
        xa = torch.cat([x, a], -1)
        qs = self.qs if only is None else [self.qs[i] for i in only]
        return torch.stack([q(xa).squeeze(-1) for q in qs])


def train(c, out):
    from vec import VecEnv
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(c.seed)
    rng = np.random.default_rng(c.seed)
    torch.set_num_threads(c.threads)
    env_cfg, rand = make_env_cfg(c)
    if c.rand_overrides:
        from bizbot_mj import P, U
        rand = replace(rand, **{k: P(v["p"]) if isinstance(v, dict) else U(*v) for k, v in c.rand_overrides.items()})
    (out / "config.json").write_text(json.dumps({"td3": asdict(c), "env": asdict(env_cfg)}, indent=1))
    venv = VecEnv(env_cfg, rand, c.workers, seed=c.seed, scale=c.scale, env_kind=c.env)
    N, od, pd = venv.n, venv.obs_size, venv.priv_size
    cd = od + (pd if c.asym else 0)
    dev = torch.device(c.device if (c.device != "mps" or torch.backends.mps.is_available()) else "cpu")
    act = ACT[c.activation]
    assert c.asym or not c.actor_priv
    psel = slice(PRIV_STATE, None) if c.actor_priv == "params" else slice(None)
    ad = od + (pd - PRIV_STATE if c.actor_priv == "params" else pd) if c.actor_priv else od
    actor, actor_t = TanhActor(ad, c.actor_hidden, act).to(dev), TanhActor(ad, c.actor_hidden, act).to(dev)
    make_critic = (lambda: Twin(cd, c.critic_hidden, act, c.critic_layernorm)) if c.n_critics <= 2 else \
        (lambda: Ensemble(cd, c.critic_hidden, act, c.critic_layernorm, c.n_critics))
    critic, critic_t = make_critic().to(dev), make_critic().to(dev)
    actor_t.load_state_dict(actor.state_dict())
    critic_t.load_state_dict(critic.state_dict())
    actor_cpu = TanhActor(ad, c.actor_hidden, act)
    actor_cpu.load_state_dict({k: v.cpu() for k, v in actor.state_dict().items()})
    opt_a = torch.optim.Adam(actor.parameters(), lr=c.lr_actor)
    opt_c = torch.optim.Adam(critic.parameters(), lr=c.lr_critic)
    onorm, pnorm = RunningNorm(od), RunningNorm(pd)
    if c.init_learner:
        L = torch.load(c.init_learner, weights_only=False)
        critic.load_state_dict(L["critic"])
        critic_t.load_state_dict(L["critic_t"])
        opt_c.load_state_dict(L["opt_c"])
        for nm, st in ((onorm, L["obs_norm"]), (pnorm, L["priv_norm"])):
            nm.mean, nm.var, nm.count = np.array(st["mean"]), np.array(st["var"]), st["count"]
    if c.init_actor:
        A = torch.load(c.init_actor, weights_only=False)
        for net in (actor, actor_t, actor_cpu):
            net.load_state_dict(A["actor"])
        if not c.init_learner:
            onorm.mean, onorm.var = np.array(A["obs_norm"]["mean"]), np.array(A["obs_norm"]["var"])

    # Replay, time-major (L rows of N robots) on the update device, raw
    # observations; normalised with the current statistics when sampled.
    # n-step returns are formed at sampling time along each robot's timeline,
    # cut at the first episode end; next_obs is the terminal obs there.
    L = max(c.buffer // N, c.nstep + 2)
    R = {"o": torch.zeros(L, N, od, device=dev), "p": torch.zeros(L, N, pd, device=dev),
         "a": torch.zeros(L, N, device=dev), "r": torch.zeros(L, N, device=dev),
         "te": torch.zeros(L, N, device=dev), "dn": torch.zeros(L, N, device=dev),
         "o2": torch.zeros(L, N, od, device=dev), "p2": torch.zeros(L, N, pd, device=dev)}
    ptr, rows = 0, 0
    gam = torch.tensor([c.gamma ** k for k in range(c.nstep + 1)], device=dev)

    def tnorm(x, nm):
        mean = torch.as_tensor(nm.mean, dtype=torch.float32, device=dev)
        std = torch.as_tensor(np.sqrt(nm.var + 1e-8), dtype=torch.float32, device=dev)
        return ((x - mean) / std).clamp(-nm.clip, nm.clip)

    def teach_in(o, ci):
        """Teacher actor input: normalised obs, then the selected part of the normalised priv."""
        return torch.cat([o, ci[:, o.shape[1]:][:, psel]], -1)

    def cin_t(o, p):
        o = tnorm(o, onorm)
        return o, (torch.cat([o, tnorm(p, pnorm)], -1) if c.asym else o)

    def sample():
        B, n = c.batch, c.nstep
        age = torch.randint(n - 1, rows, (B,), device=dev)
        env = torch.randint(0, N, (B,), device=dev)
        t0 = (ptr - 1 - age) % L
        tk = (t0[:, None] + torch.arange(n, device=dev)[None, :]) % L          # (B, n)
        r = R["r"][tk, env[:, None]]
        dn = R["dn"][tk, env[:, None]]
        te = R["te"][tk, env[:, None]]
        alive = torch.cumprod(torch.cat([torch.ones(B, 1, device=dev), 1.0 - dn[:, :-1]], 1), 1)
        G = (alive * r * gam[None, :n]).sum(1)
        m = alive.sum(1).long()                                                   # steps included
        last = tk[torch.arange(B, device=dev), m - 1]
        o, ci = cin_t(R["o"][t0, env], R["p"][t0, env])
        o2, c2 = cin_t(R["o2"][last, env], R["p2"][last, env])
        d = te[torch.arange(B, device=dev), m - 1]
        return o, ci, R["a"][t0, env], G, d, o2, c2, gam[m]

    obs, priv = venv.observe()
    onorm.update(obs)
    pnorm.update(priv)
    noise = rng.uniform(c.noise_min, c.noise_max, N)
    log = open(out / "progress.csv", "w", newline="")
    writer = None
    start = time.time()
    steps, it, next_ckpt, n_upd = 0, 0, c.ckpt_every, 0
    win = np.zeros(7)
    ql, al = [], []
    t_env = t_upd = 0.0
    while time.time() - start < c.budget_s and steps < c.max_steps:
        if c.curriculum_s > 0:
            s = min(1.0, c.scale + (1.0 - c.scale) * (time.time() - start) / c.curriculum_s)
            if abs(s - venv.scale) > 0.02:
                venv.scale = s
        t0 = time.time()
        if steps < c.learning_starts:
            a = rng.uniform(-0.3, 0.3, N).astype(np.float32)
        else:
            with torch.no_grad():
                x = np.concatenate([onorm(obs), pnorm(priv)[:, psel]], 1) if c.actor_priv else onorm(obs)
                a = actor_cpu(torch.from_numpy(x)).squeeze(-1).numpy()
            a = np.clip(a + noise * rng.standard_normal(N), -1.0, 1.0).astype(np.float32)
        obs2, priv2, rew, term, trunc, tobs, tpriv, st = venv.step(a)
        win += st
        done = term | trunc
        nxt_o, nxt_p = obs2.copy(), priv2.copy()
        if done.any():
            di = np.nonzero(done)[0]
            nxt_o[di], nxt_p[di] = tobs[di], tpriv[di]
            noise[di] = rng.uniform(c.noise_min, c.noise_max, di.size)
        T_ = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(dev)
        R["o"][ptr], R["p"][ptr], R["a"][ptr], R["r"][ptr] = T_(obs), T_(priv), T_(a), T_(rew)
        R["te"][ptr], R["dn"][ptr], R["o2"][ptr], R["p2"][ptr] = T_(term), T_(done), T_(nxt_o), T_(nxt_p)
        ptr = (ptr + 1) % L
        rows = min(L, rows + 1)
        obs, priv = obs2, priv2
        onorm.update(obs)
        pnorm.update(priv)
        steps += N
        t_env += time.time() - t0

        t0 = time.time()
        if steps >= c.learning_starts and rows > c.nstep + 1:
            for _ in range(c.updates_per_step):
                o, ci, a_b, r, d, o2, c2, g = sample()
                with torch.no_grad():
                    na = actor_t(teach_in(o2, c2) if c.actor_priv else o2).squeeze(-1)
                    na = (na + (torch.randn_like(na) * c.target_noise).clamp(-c.target_clip, c.target_clip)).clamp(-1, 1)
                    if c.n_critics <= 2:
                        q1t, q2t = critic_t(c2, na.unsqueeze(-1))
                        qt = torch.min(q1t, q2t)
                    else:
                        pick = torch.randperm(c.n_critics)[:2].tolist()
                        qt = torch.min(critic_t(c2, na.unsqueeze(-1), only=pick), 0).values
                    y = r + g * (1.0 - d) * qt
                if c.n_critics <= 2:
                    q1, q2 = critic(ci, a_b.unsqueeze(-1))
                    lq = ((q1 - y) ** 2).mean() + ((q2 - y) ** 2).mean()
                else:
                    lq = ((critic(ci, a_b.unsqueeze(-1)) - y) ** 2).mean(1).sum()
                opt_c.zero_grad()
                lq.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
                opt_c.step()
                n_upd += 1
                if n_upd % c.policy_delay == 0 and n_upd >= c.actor_freeze_updates:
                    pa = actor(teach_in(o, ci) if c.actor_priv else o)
                    la = -(critic(ci, pa)[0] if c.n_critics <= 2 else critic(ci, pa).mean(0)).mean()
                    opt_a.zero_grad()
                    la.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
                    opt_a.step()
                    with torch.no_grad():
                        for p_, pt in zip(actor.parameters(), actor_t.parameters()):
                            pt.mul_(1 - c.tau).add_(c.tau * p_)
                        for p_, pt in zip(critic.parameters(), critic_t.parameters()):
                            pt.mul_(1 - c.tau).add_(c.tau * p_)
                    al.append(la.detach())
                ql.append(lq.detach())
            actor_cpu.load_state_dict({k: v.cpu() for k, v in actor.state_dict().items()})
        t_upd += time.time() - t0
        it += 1

        if it % 100 == 0 or time.time() - start >= c.budget_s:
            done_, fell, rsum, lsum, v2, r2, alive_ = win
            row = {"steps": steps, "time": round(time.time() - start, 1), "fps": int(steps / (time.time() - start)),
                   "scale": round(venv.scale, 3), "fell_frac": fell / done_ if done_ else np.nan,
                   "ep_return": rsum / done_ if done_ else np.nan, "ep_len": lsum / done_ if done_ else np.nan,
                   "speed_rms": np.sqrt(v2 / max(alive_, 1)), "rate_rms_dps": np.degrees(np.sqrt(r2 / max(alive_, 1))),
                   "falls_per_robot_min": fell / max(alive_, 1) * (60 / env_cfg.policy_dt),
                   "q_loss": float(torch.stack(ql).mean()) if ql else np.nan,
                   "a_loss": float(torch.stack(al).mean()) if al else np.nan,
                   "buffer": rows * N, "t_env": round(t_env, 1), "t_upd": round(t_upd, 1)}
            row = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in row.items()}
            ql, al = [], []
            win[:] = 0
            if writer is None:
                writer = csv.DictWriter(log, fieldnames=list(row))
                writer.writeheader()
            writer.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in row.items()})
            log.flush()
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        if steps >= next_ckpt:
            save(out / f"ckpt_{steps // 1_000_000:04d}M.pt", actor_cpu, onorm, c, env_cfg, pnorm)
            # Full state, overwritten each time, so a crash late in a long run loses little.
            save_learner(out / "latest.learner.pt", c, env_cfg, actor=actor, actor_t=actor_t, critic=critic,
                         critic_t=critic_t, opt_a=opt_a, opt_c=opt_c, onorm=onorm, pnorm=pnorm)
            next_ckpt += c.ckpt_every
    save(out / "final.pt", actor_cpu, onorm, c, env_cfg, pnorm)
    save_learner(out / "final.learner.pt", c, env_cfg, actor=actor, actor_t=actor_t, critic=critic,
                 critic_t=critic_t, opt_a=opt_a, opt_c=opt_c, onorm=onorm, pnorm=pnorm)
    venv.close()
    log.close()


def save(path, actor, onorm, c, env_cfg, pnorm=None):
    torch.save({"kind": "td3", "actor": actor.state_dict(), "obs_norm": onorm.state(), "td3": asdict(c),
                "env": asdict(env_cfg), **({"priv_norm": pnorm.state()} if pnorm is not None else {})}, path)


def save_learner(path, c, env_cfg, onorm, pnorm, **parts):
    """Everything needed to keep training: networks, target networks, optimiser state and normalisers.
    (The replay buffer is not saved; a fine-tune refills it.)"""
    torch.save({"kind": "td3-learner", "td3": asdict(c), "env": asdict(env_cfg),
                "obs_norm": {**onorm.state(), "count": onorm.count},
                "priv_norm": {**pnorm.state(), "count": pnorm.count},
                **{k: {n: t.detach().cpu() if torch.is_tensor(t) else t for n, t in v.state_dict().items()}
                   if hasattr(v, "state_dict") else v for k, v in parts.items()}}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--json", type=str, default="{}")
    args = ap.parse_args()
    over = json.loads(args.json)
    for f in fields(TD3Config):
        if f.name in over and isinstance(getattr(TD3Config, f.name, None), tuple):
            over[f.name] = tuple(over[f.name])
    train(TD3Config(**over), args.out)


if __name__ == "__main__":
    main()
