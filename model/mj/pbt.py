"""Population-based training of TD3 on the MuJoCo sim.

    .venv/bin/python -u model/mj/pbt.py --out model/mj/runs/pbt1 --json '{"total_s": 28800}'

`members` TD3 learners share one set of simulated robots, each controlling its
own block, with its own networks, optimisers, observation normaliser and
replay buffer. Every `generation_s` every member's actor is scored
deterministically on fixed robots (seed changes each generation). The worst
`exploit` share copy the networks, optimisers, normaliser and hyperparameters
of a random member from the best share, keep their own replay buffer, and
multiply each tunable hyperparameter by 0.8 or 1.25. Members with
`soft_reset` shrink their weights toward a fresh initialisation (w <- a w +
(1 - a) w_init) right after each selection. Rewards are fixed; selection uses
only the evaluation score.

Writes pbt.csv (one row per member per generation), members/<k>/ckpt_*.pt
every ckpt_every samples of that member, gen_<g>/m<k>.pt at every
generation, and best.pt (the best-scoring actor so far).
"""

import argparse
import copy
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(HERE))
from ppo import ACT, RunningNorm  # noqa: E402
from td3 import TanhActor, TD3Config, Twin, save, save_learner  # noqa: E402

TUNABLE = {  # name: (low, high)
    "lr_actor": (3e-5, 3e-3), "lr_critic": (3e-5, 3e-3), "tau": (0.005, 0.2),
    "noise_max": (0.03, 0.5), "target_noise": (0.02, 0.3),
}


@dataclass
class PBTConfig:
    members: int = 8
    workers: int = 12
    per_worker: int = 256
    batch: int = 2048
    updates_per_step: int = 2
    buffer: int = 500_000            # per member
    learning_starts: int = 25_000    # per member samples
    generation_s: float = 1800.0
    total_s: float = 8 * 3600.0
    exploit: float = 0.25
    soft_reset_alpha: float = 0.8
    eval_n: int = 96
    eval_rows: tuple = ("nominal", "50% ranges", "full ranges")
    eval_procs: int = 12
    ckpt_every: int = 10_000_000     # per member samples
    scale: float = 1.0
    td3: dict = field(default_factory=dict)          # TD3Config overrides shared by all members
    env_overrides: dict = field(default_factory=dict)
    seed: int = 0
    device: str = "mps"


# --- fitness ---------------------------------------------------------------------
# Lower is better. Falls dominate: 10 points more falls on nominal or 50%
# ranges (5 on full) cost more than all of that row's wobble and drift. Each
# quality term is relative to roughly what the per-robot LQR gets, and a row
# where every robot fell counts its quality terms at three times that reference.
COST = {"nominal": dict(fell=10.0, speed=(0.25, 0.15), rate=(0.25, 4.0), creep=(0.1, 0.02)),
        "50% ranges": dict(fell=10.0, speed=(0.5, 0.4), rate=(0.5, 30.0), creep=(0.25, 0.12)),
        "full ranges": dict(fell=5.0, speed=(0.25, 1.0), rate=(0.25, 60.0))}


def cost(results):
    total = 0.0
    for row, r in results.items():
        w = COST[row]
        total += w["fell"] * r["fell"]
        for k in ("speed", "rate", "creep"):
            if k not in w:
                continue
            wk, ref = w[k]
            v = r[k] if np.isfinite(r[k]) else 3.0 * ref
            total += wk * v / ref
    return total


def _eval_job(job):
    path, row, n, seed = job
    import metrics
    from ppo import load_policy
    policy, ck = load_policy(path)
    env = ck["env"]
    rand, scale = metrics.ROWS[row]
    r = metrics.run(lambda m, obs: policy(obs), rand, scale, n=n, seconds=20.0, seed=seed,
                    policy_dt=env["policy_dt"],
                    env_kwargs={k: v for k, v in env.items() if k not in ("n_envs", "policy_dt", "episode_s")})
    return path, row, r


# --- one TD3 learner -------------------------------------------------------------

class Member:
    def __init__(self, k, sl, od, pd, cfg, td3c, dev, rng):
        self.k, self.sl, self.od, self.pd, self.dev, self.rng = k, sl, od, pd, dev, rng
        self.N = sl.stop - sl.start
        self.c = td3c
        self.hp = {name: float(getattr(td3c, name)) for name in TUNABLE}
        self.soft_reset = bool(k % 2)
        self.cfg = cfg
        act = ACT[td3c.activation]
        self.cd = od + (pd if td3c.asym else 0)
        self.actor = TanhActor(od, td3c.actor_hidden, act).to(dev)
        self.actor_t = copy.deepcopy(self.actor)
        self.critic = Twin(self.cd, td3c.critic_hidden, act, td3c.critic_layernorm).to(dev)
        self.critic_t = copy.deepcopy(self.critic)
        self.actor_cpu = TanhActor(od, td3c.actor_hidden, act)
        self._sync_cpu()
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=self.hp["lr_actor"])
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=self.hp["lr_critic"])
        self.onorm, self.pnorm = RunningNorm(od), RunningNorm(pd)
        L = max(cfg.buffer // self.N, td3c.nstep + 2)
        self.L = L
        z = lambda *s: torch.zeros(*s, device=dev)
        self.R = {"o": z(L, self.N, od), "p": z(L, self.N, pd), "a": z(L, self.N), "r": z(L, self.N),
                  "te": z(L, self.N), "dn": z(L, self.N), "o2": z(L, self.N, od), "p2": z(L, self.N, pd)}
        self.ptr = self.rows = 0
        self.gam = torch.tensor([td3c.gamma ** i for i in range(td3c.nstep + 1)], device=dev)
        self.noise = rng.uniform(td3c.noise_min, self.hp["noise_max"], self.N)
        self.samples = self.n_upd = self.since_sync = 0
        self.next_ckpt = cfg.ckpt_every
        self.win = np.zeros(3)   # falls, finished episodes, robot-steps

    def _sync_cpu(self):
        self.actor_cpu.load_state_dict({k: v.cpu() for k, v in self.actor.state_dict().items()})

    def set_lr(self):
        for g in self.opt_a.param_groups:
            g["lr"] = self.hp["lr_actor"]
        for g in self.opt_c.param_groups:
            g["lr"] = self.hp["lr_critic"]

    def act(self, obs):
        if self.samples < self.cfg.learning_starts:
            return self.rng.uniform(-0.3, 0.3, self.N).astype(np.float32)
        with torch.no_grad():
            a = self.actor_cpu(torch.from_numpy(self.onorm(obs))).squeeze(-1).numpy()
        return np.clip(a + self.noise * self.rng.standard_normal(self.N), -1.0, 1.0).astype(np.float32)

    def store(self, obs, priv, a, rew, term, done, nxt_o, nxt_p):
        R, p, od, pd = self.R, self.ptr, self.od, self.pd
        col = lambda x: np.asarray(x, dtype=np.float32)[:, None]
        packed = torch.from_numpy(np.concatenate([obs, priv, col(a), col(rew), col(term), col(done), nxt_o, nxt_p],
                                                 1).astype(np.float32)).to(self.dev)
        R["o"][p], R["p"][p] = packed[:, :od], packed[:, od:od + pd]
        k = od + pd
        R["a"][p], R["r"][p], R["te"][p], R["dn"][p] = packed[:, k], packed[:, k + 1], packed[:, k + 2], packed[:, k + 3]
        R["o2"][p], R["p2"][p] = packed[:, k + 4:k + 4 + od], packed[:, k + 4 + od:]
        self.ptr = (p + 1) % self.L
        self.rows = min(self.L, self.rows + 1)
        di = np.nonzero(done)[0]
        if di.size:
            self.noise[di] = self.rng.uniform(self.c.noise_min, self.hp["noise_max"], di.size)
        self.onorm.update(obs)
        self.pnorm.update(priv)
        self.samples += self.N
        self.win += (term.sum(), done.sum(), self.N - term.sum())

    def _tnorm(self, x, nm):
        mean = torch.as_tensor(nm.mean, dtype=torch.float32, device=self.dev)
        std = torch.as_tensor(np.sqrt(nm.var + 1e-8), dtype=torch.float32, device=self.dev)
        return ((x - mean) / std).clamp(-nm.clip, nm.clip)

    def _cin(self, o, p):
        o = self._tnorm(o, self.onorm)
        return o, (torch.cat([o, self._tnorm(p, self.pnorm)], -1) if self.c.asym else o)

    def _sample(self):
        B, n, dev, R = self.cfg.batch, self.c.nstep, self.dev, self.R
        age = torch.randint(n - 1, self.rows, (B,), device=dev)
        env = torch.randint(0, self.N, (B,), device=dev)
        t0 = (self.ptr - 1 - age) % self.L
        tk = (t0[:, None] + torch.arange(n, device=dev)[None, :]) % self.L
        r, dn, te = R["r"][tk, env[:, None]], R["dn"][tk, env[:, None]], R["te"][tk, env[:, None]]
        alive = torch.cumprod(torch.cat([torch.ones(B, 1, device=dev), 1.0 - dn[:, :-1]], 1), 1)
        G = (alive * r * self.gam[None, :n]).sum(1)
        m = alive.sum(1).long()
        ar = torch.arange(B, device=dev)
        last = tk[ar, m - 1]
        o, ci = self._cin(R["o"][t0, env], R["p"][t0, env])
        o2, c2 = self._cin(R["o2"][last, env], R["p2"][last, env])
        return o, ci, R["a"][t0, env], G, te[ar, m - 1], o2, c2, self.gam[m]

    def update(self):
        if self.samples < self.cfg.learning_starts or self.rows <= self.c.nstep + 1:
            return
        hp = self.hp
        for _ in range(self.cfg.updates_per_step):
            o, ci, a_b, r, d, o2, c2, g = self._sample()
            with torch.no_grad():
                na = self.actor_t(o2).squeeze(-1)
                tn = hp["target_noise"]
                na = (na + (torch.randn_like(na) * tn).clamp(-2 * tn, 2 * tn)).clamp(-1, 1)
                q1t, q2t = self.critic_t(c2, na.unsqueeze(-1))
                y = r + g * (1.0 - d) * torch.min(q1t, q2t)
            q1, q2 = self.critic(ci, a_b.unsqueeze(-1))
            lq = ((q1 - y) ** 2).mean() + ((q2 - y) ** 2).mean()
            self.opt_c.zero_grad()
            lq.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
            self.opt_c.step()
            self.n_upd += 1
            if self.n_upd % self.c.policy_delay == 0:
                la = -self.critic(ci, self.actor(o))[0].mean()
                self.opt_a.zero_grad()
                la.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
                self.opt_a.step()
                tau = hp["tau"]
                with torch.no_grad():
                    src = list(self.actor.parameters()) + list(self.critic.parameters())
                    tgt = list(self.actor_t.parameters()) + list(self.critic_t.parameters())
                    torch._foreach_mul_(tgt, 1.0 - tau)
                    torch._foreach_add_(tgt, src, alpha=tau)
        # Copying the actor back to the CPU stalls the GPU; every 10 steps is plenty off-policy.
        self.since_sync += 1
        if self.since_sync >= 10:
            self._sync_cpu()
            self.since_sync = 0

    def td3_cfg(self):
        return replace(self.c, **{k: (v if k != "noise_max" else v) for k, v in self.hp.items()})

    def save_actor(self, path, env_cfg):
        self._sync_cpu()
        save(path, self.actor_cpu, self.onorm, self.td3_cfg(), env_cfg)

    def save_full(self, path, env_cfg):
        save_learner(path, self.td3_cfg(), env_cfg, onorm=self.onorm, pnorm=self.pnorm, actor=self.actor,
                     actor_t=self.actor_t, critic=self.critic, critic_t=self.critic_t, opt_a=self.opt_a,
                     opt_c=self.opt_c, hp=dict(self.hp), soft_reset=self.soft_reset, samples=self.samples)

    def copy_from(self, other):
        for mine, theirs in ((self.actor, other.actor), (self.actor_t, other.actor_t),
                             (self.critic, other.critic), (self.critic_t, other.critic_t)):
            mine.load_state_dict(theirs.state_dict())
        self.opt_a.load_state_dict(other.opt_a.state_dict())
        self.opt_c.load_state_dict(other.opt_c.state_dict())
        for nm, on in ((self.onorm, other.onorm), (self.pnorm, other.pnorm)):
            nm.mean, nm.var, nm.count = on.mean.copy(), on.var.copy(), on.count
        self.hp = dict(other.hp)
        self.soft_reset = other.soft_reset
        self._sync_cpu()

    def explore(self):
        for k, (lo, hi) in TUNABLE.items():
            self.hp[k] = float(np.clip(self.hp[k] * self.rng.choice([0.8, 1.25]), lo, hi))
        if self.rng.random() < 0.2:
            self.soft_reset = not self.soft_reset
        self.set_lr()

    def shrink_perturb(self):
        a = self.cfg.soft_reset_alpha
        act = ACT[self.c.activation]
        fresh = {"actor": TanhActor(self.od, self.c.actor_hidden, act).to(self.dev),
                 "critic": Twin(self.cd, self.c.critic_hidden, act, self.c.critic_layernorm).to(self.dev)}
        with torch.no_grad():
            for name, net in (("actor", self.actor), ("critic", self.critic)):
                for p_, f in zip(net.parameters(), fresh[name].parameters()):
                    p_.mul_(a).add_((1 - a) * f)
        self._sync_cpu()


# --- driver -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--json", type=str, default="{}")
    args = ap.parse_args()
    over = json.loads(args.json)
    if "eval_rows" in over:
        over["eval_rows"] = tuple(over["eval_rows"])
    cfg = PBTConfig(**over)
    td3_over = dict(cfg.td3)
    for k in ("actor_hidden", "critic_hidden"):
        if k in td3_over:
            td3_over[k] = tuple(td3_over[k])
    td3c = TD3Config(**td3_over)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    from bizbot_mj import Randomization
    from bizbot_mj.env import EnvConfig
    from vec import VecEnv
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    env_cfg = EnvConfig(n_envs=cfg.per_worker, **cfg.env_overrides)
    (out / "config.json").write_text(json.dumps({"pbt": asdict(cfg), "td3": asdict(td3c), "env": asdict(env_cfg)},
                                                indent=1))
    venv = VecEnv(env_cfg, Randomization(), cfg.workers, seed=cfg.seed, scale=cfg.scale)
    N, od, pd = venv.n, venv.obs_size, venv.priv_size
    dev = torch.device(cfg.device if (cfg.device != "mps" or torch.backends.mps.is_available()) else "cpu")
    per = N // cfg.members
    members = []
    for k in range(cfg.members):
        torch.manual_seed(cfg.seed * 100 + k)
        m = Member(k, slice(k * per, (k + 1) * per), od, pd, cfg, td3c, dev, np.random.default_rng(cfg.seed * 100 + k))
        if k > 0:   # start from a spread of hyperparameters
            for _ in range(2):
                m.explore()
        members.append(m)
        (out / "members" / str(k)).mkdir(parents=True, exist_ok=True)

    log_f = open(out / "pbt.csv", "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["gen", "time_h", "member", "samples", "updates", "cost", "rank", "action", "soft_reset"]
                 + list(TUNABLE) + [f"{row}|{k}" for row in cfg.eval_rows for k in ("fell", "speed", "rate", "hf", "creep")])
    best_cost = np.inf
    obs, priv = venv.observe()
    start = gen_start = time.time()
    gen, steps = 0, 0
    pool = ProcessPoolExecutor(max_workers=cfg.eval_procs)
    while time.time() - start < cfg.total_s:
        acts = np.concatenate([m.act(obs[m.sl]) for m in members] + [np.zeros(N - per * cfg.members, np.float32)])
        obs2, priv2, rew, term, trunc, tobs, tpriv, _ = venv.step(acts)
        done = term | trunc
        nxt_o, nxt_p = obs2.copy(), priv2.copy()
        if done.any():
            di = np.nonzero(done)[0]
            nxt_o[di], nxt_p[di] = tobs[di], tpriv[di]
        for m in members:
            s = m.sl
            m.store(obs[s], priv[s], acts[s], rew[s], term[s], done[s], nxt_o[s], nxt_p[s])
            m.update()
            if m.samples >= m.next_ckpt:
                m.save_actor(out / "members" / str(m.k) / f"ckpt_{m.samples // 1_000_000:05d}M.pt", env_cfg)
                m.next_ckpt += cfg.ckpt_every
        obs, priv = obs2, priv2
        steps += 1
        if steps % 2000 == 0:
            el = time.time() - start
            print(json.dumps({"t_h": round(el / 3600, 3), "gen": gen, "fps": int(steps * N / el),
                              "falls_per_robot_min": [round(float(m.win[0] / max(m.win[2], 1) * 6000), 3) for m in members]}),
                  flush=True)
            for m in members:
                m.win[:] = 0

        if time.time() - gen_start >= cfg.generation_s or time.time() - start >= cfg.total_s:
            gdir = out / f"gen_{gen:03d}"
            gdir.mkdir(exist_ok=True)
            paths = []
            for m in members:
                p = gdir / f"m{m.k}.pt"
                m.save_actor(p, env_cfg)
                paths.append(str(p))
            jobs = [(p, row, cfg.eval_n, 5000 + gen) for p in paths for row in cfg.eval_rows]
            res = {}
            for p, row, r in pool.map(_eval_job, jobs):
                res.setdefault(p, {})[row] = r
            costs = np.array([cost(res[p]) for p in paths])
            order = np.argsort(costs)
            n_ex = max(1, int(round(cfg.exploit * cfg.members)))
            top, bottom = order[:n_ex], order[-n_ex:]
            actions = {k: "keep" for k in range(cfg.members)}
            # Full learner state of this generation's top member, for fine-tuning later.
            members[order[0]].save_full(gdir / f"m{int(order[0])}.learner.pt", env_cfg)
            if costs[order[0]] < best_cost:
                best_cost = costs[order[0]]
                members[order[0]].save_actor(out / "best.pt", env_cfg)
            for k in range(cfg.members):
                m = members[k]
                row = [gen, round((time.time() - start) / 3600, 3), k, m.samples, m.n_upd, round(float(costs[k]), 4),
                       int(np.nonzero(order == k)[0][0]), "", int(m.soft_reset)] + [m.hp[t] for t in TUNABLE]
                row += [round(float(res[paths[k]][rw][q]), 4) for rw in cfg.eval_rows for q in ("fell", "speed", "rate", "hf", "creep")]
                log_rows = row
                if k in bottom and gen < 10 ** 6:
                    src = int(rng.choice(top))
                    m.copy_from(members[src])
                    m.explore()
                    actions[k] = f"copy m{src}"
                log_rows[7] = actions[k]
                log.writerow(log_rows)
            log_f.flush()
            for m in members:
                if m.soft_reset:
                    m.shrink_perturb()
            print(json.dumps({"gen": gen, "costs": [round(float(c), 3) for c in costs], "best": int(order[0]),
                              "best_cost": round(float(best_cost), 3), "actions": actions}), flush=True)
            gen += 1
            gen_start = time.time()
    pool.shutdown()
    venv.close()
    log_f.close()


if __name__ == "__main__":
    main()
