"""PPO for the balance task, with an optional privileged critic.

    .venv/bin/python -u model/mj/ppo.py --out model/mj/runs/test --budget-s 300 --workers 6

The actor sees only the ESP32's observation. With --asym the critic also sees
the true state and the robot's parameters (training only). Stops at the time
budget or step budget, whichever first. Writes progress.csv, config.json and
checkpoints (actor + observation normaliser) every --ckpt-every steps.
"""

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(HERE))


@dataclass
class PPOConfig:
    env: str = "mj"                  # "mj" (MuJoCo) or "old" (sim/bizbot_sim)
    workers: int = 6
    per_worker: int = 256
    n_steps: int = 32
    epochs: int = 5
    minibatches: int = 4
    lr: float = 3e-4
    lr_schedule: str = "adaptive"    # "adaptive" (KL-targeted) or "constant"
    target_kl: float = 0.01
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 1.0
    max_grad_norm: float = 1.0
    log_std_init: float = -1.0
    actor_hidden: tuple = (128, 128)
    critic_hidden: tuple = (256, 256)
    activation: str = "elu"
    asym: bool = True                # privileged critic
    recurrent: bool = False          # GRU actor over the observation stream
    rnn_hidden: int = 64
    norm_obs: bool = True
    norm_reward: bool = True
    scale: float = 1.0               # randomization width; with curriculum, the start
    curriculum_s: float = 0.0        # seconds of wall time to ramp scale to 1 (0 = fixed)
    budget_s: float = 300.0
    max_steps: int = 10 ** 10
    ckpt_every: int = 5_000_000
    threads: int = 2
    device: str = "mps"              # where updates run; rollouts use CPU copies
    seed: int = 0
    env_overrides: dict = field(default_factory=dict)


ACT = {"elu": nn.ELU, "tanh": nn.Tanh, "relu": nn.ReLU}


def mlp(i, hidden, o, act):
    layers, d = [], i
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class RunningNorm:
    def __init__(self, size, clip=10.0):
        self.mean, self.var, self.count, self.clip = np.zeros(size), np.ones(size), 1e-4, clip

    def update(self, x):
        x = x.reshape(-1, x.shape[-1]).astype(np.float64)
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        delta, tot = bm - self.mean, self.count + bc
        self.mean = self.mean + delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -self.clip, self.clip).astype(np.float32)

    def state(self):
        return {"mean": self.mean.tolist(), "var": self.var.tolist()}


class Actor(nn.Module):
    def __init__(self, obs, hidden, act, log_std_init):
        super().__init__()
        self.net = mlp(obs, hidden, 1, act)
        self.log_std = nn.Parameter(torch.full((1,), float(log_std_init)))

    def dist(self, x):
        mu = self.net(x)
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))


class RecActor(nn.Module):
    """GRU over the observation stream, then an MLP head. reset[t] zeroes the
    memory of robots whose episode restarted before step t."""

    def __init__(self, obs, rnn_hidden, hidden, act, log_std_init):
        super().__init__()
        self.cell = nn.GRUCell(obs, rnn_hidden)
        self.net = mlp(rnn_hidden, hidden, 1, act)
        self.log_std = nn.Parameter(torch.full((1,), float(log_std_init)))
        self.rnn_hidden = rnn_hidden

    def step(self, x, h, reset):
        h = self.cell(x, h * (1.0 - reset).unsqueeze(-1))
        mu = self.net(h)
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu)), h

    def seq(self, x, h0, reset):
        h, mus = h0, []
        for t in range(x.shape[0]):
            h = self.cell(x[t], h * (1.0 - reset[t]).unsqueeze(-1))
            mus.append(self.net(h))
        mu = torch.stack(mus)
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))


def make_actor(c, od, act):
    if c.recurrent:
        return RecActor(od, c.rnn_hidden, c.actor_hidden, act, c.log_std_init)
    return Actor(od, c.actor_hidden, act, c.log_std_init)


def make_env_cfg(c):
    if c.env == "mj":
        from bizbot_mj.env import EnvConfig
        from bizbot_mj import Randomization
    else:
        from bizbot_sim import EnvConfig, Randomization
    return EnvConfig(n_envs=c.per_worker, **c.env_overrides), Randomization()


def train(c, out):
    from vec import VecEnv
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(c.seed)
    np.random.seed(c.seed)
    torch.set_num_threads(c.threads)
    env_cfg, rand = make_env_cfg(c)
    (out / "config.json").write_text(json.dumps({"ppo": asdict(c), "env": asdict(env_cfg)}, indent=1))
    venv = VecEnv(env_cfg, rand, c.workers, seed=c.seed, scale=c.scale, env_kind=c.env)
    N, od, pd = venv.n, venv.obs_size, venv.priv_size
    act = ACT[c.activation]
    dev = torch.device(c.device if (c.device != "mps" or torch.backends.mps.is_available()) else "cpu")
    actor_cpu = make_actor(c, od, act)
    critic_cpu = mlp(od + (pd if c.asym else 0), c.critic_hidden, 1, act)
    actor = make_actor(c, od, act).to(dev)
    critic = mlp(od + (pd if c.asym else 0), c.critic_hidden, 1, act).to(dev)
    actor.load_state_dict(actor_cpu.state_dict())
    critic.load_state_dict(critic_cpu.state_dict())
    params = list(actor.parameters()) + list(critic.parameters())
    opt = torch.optim.Adam(params, lr=c.lr)
    lr = c.lr
    onorm, pnorm = RunningNorm(od), RunningNorm(pd)
    if not c.norm_obs:
        onorm.update = pnorm.update = lambda x: None
    ret_rms, ret_acc = RunningNorm(1), np.zeros(N)

    def c_in(o, p):
        return np.concatenate([o, p], axis=1) if c.asym else o

    T = c.n_steps
    buf = {k: np.zeros((T, N) + s, np.float32) for k, s in
           (("obs", (od,)), ("cin", (od + (pd if c.asym else 0),)), ("act", ()), ("logp", ()), ("val", ()),
            ("rew", ()), ("done", ()), ("boot", ()))}
    obs, priv = venv.observe()
    onorm.update(obs)
    pnorm.update(priv)
    if c.recurrent:
        h = torch.zeros(N, c.rnn_hidden)
        prev_done = np.zeros(N, np.float32)
        buf["reset"] = np.zeros((T, N), np.float32)
    log = open(out / "progress.csv", "w", newline="")
    writer = None
    start = time.time()
    steps, it, next_ckpt = 0, 0, c.ckpt_every
    win = np.zeros(7)
    while time.time() - start < c.budget_s and steps < c.max_steps:
        if c.curriculum_s > 0:
            s = min(1.0, c.scale + (1.0 - c.scale) * (time.time() - start) / c.curriculum_s)
            if abs(s - venv.scale) > 0.02:
                venv.scale = s
        t_roll = time.time()
        raw_obs, raw_priv, trunc_list = [], [], []
        if c.recurrent:
            h0 = h.clone()
        for t in range(T):
            o_n, p_n = onorm(obs), pnorm(priv)
            ci = c_in(o_n, p_n)
            with torch.no_grad():
                if c.recurrent:
                    buf["reset"][t] = prev_done
                    d, h = actor_cpu.step(torch.from_numpy(o_n), h, torch.from_numpy(prev_done))
                else:
                    d = actor_cpu.dist(torch.from_numpy(o_n))
                a = d.sample()
                logp = d.log_prob(a).squeeze(-1).numpy()
            a = a.squeeze(-1).numpy()
            obs, priv, rew, term, trunc, tobs, tpriv, st = venv.step(np.clip(a, -1.0, 1.0))
            raw_obs.append(obs)
            raw_priv.append(priv)
            if trunc.any():
                ti = np.nonzero(trunc)[0]
                trunc_list.append((t, ti, c_in(onorm(tobs[ti]), pnorm(tpriv[ti]))))
            v = 0.0
            if c.norm_reward:
                ret_acc = ret_acc * c.gamma + rew
                ret_rms.update(ret_acc[:, None])
                ret_acc[term | trunc] = 0.0
                rew = rew / np.sqrt(ret_rms.var[0] + 1e-8)
            buf["obs"][t], buf["cin"][t], buf["act"][t], buf["logp"][t], buf["val"][t] = o_n, ci, a, logp, v
            buf["rew"][t], buf["done"][t] = rew, term | trunc
            if c.recurrent:
                prev_done = (term | trunc).astype(np.float32)
            win += st
        t_roll = time.time() - t_roll
        # Values for the whole rollout in one batch on the update device.
        with torch.no_grad():
            V = lambda x: critic(torch.from_numpy(np.ascontiguousarray(x)).to(dev)).squeeze(-1).cpu().numpy()
            buf["val"][:] = V(buf["cin"].reshape(T * N, -1)).reshape(T, N)
            last_v = V(c_in(onorm(obs), pnorm(priv)))
            buf["boot"][:] = 0.0
            for t, ti, ci_t in trunc_list:
                buf["boot"][t, ti] = V(ci_t)
        onorm.update(np.stack(raw_obs))
        pnorm.update(np.stack(raw_priv))
        steps += T * N

        # GAE, bootstrapping truncated episodes from their final observation.
        adv = np.zeros((T, N), np.float32)
        gae, nv = np.zeros(N, np.float32), last_v
        for t in reversed(range(T)):
            nd = 1.0 - buf["done"][t]
            delta = buf["rew"][t] + c.gamma * buf["boot"][t] + c.gamma * nv * nd - buf["val"][t]
            gae = delta + c.gamma * c.lam * nd * gae
            adv[t] = gae
            nv = buf["val"][t]
        ret = adv + buf["val"]

        t_upd = time.time()
        B = T * N
        flat = {k: torch.from_numpy(v.reshape(B, *v.shape[2:])).to(dev) for k, v in buf.items()}
        f_adv, f_ret = torch.from_numpy(adv.reshape(B)).to(dev), torch.from_numpy(ret.reshape(B)).to(dev)
        mb = B // c.minibatches
        kls, clips, vls, pls, ep_stats = [], [], [], [], []
        stop = False
        if c.recurrent:
            seqs = {k: torch.from_numpy(buf[k]).to(dev) for k in ("obs", "reset", "act")}
            h0_d = h0.to(dev)
            idx_tn = torch.arange(B, device=dev).reshape(T, N)
        for ep in range(c.epochs):
            perm = torch.randperm(B, device=dev)
            eperm = torch.randperm(N, device=dev)
            for k in range(c.minibatches):
                if c.recurrent:
                    cols = eperm[k * (N // c.minibatches):(k + 1) * (N // c.minibatches)]
                    idx = idx_tn[:, cols].reshape(-1)
                    d = actor.seq(seqs["obs"][:, cols], h0_d[cols], seqs["reset"][:, cols])
                    logp = d.log_prob(seqs["act"][:, cols].unsqueeze(-1)).squeeze(-1).reshape(-1)
                else:
                    idx = perm[k * mb:(k + 1) * mb]
                    d = actor.dist(flat["obs"][idx])
                    logp = d.log_prob(flat["act"][idx].unsqueeze(-1)).squeeze(-1)
                ratio = torch.exp(logp - flat["logp"][idx])
                A = f_adv[idx]
                A = (A - A.mean()) / (A.std() + 1e-8)
                pl = -torch.min(ratio * A, ratio.clamp(1 - c.clip, 1 + c.clip) * A).mean()
                v = critic(flat["cin"][idx]).squeeze(-1)
                vl = ((v - f_ret[idx]) ** 2).mean()
                ent = d.entropy().mean()
                loss = pl + c.vf_coef * vl - c.ent_coef * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, c.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    lr_ = logp - flat["logp"][idx]
                    ep_stats.append(torch.stack([((torch.exp(lr_) - 1) - lr_).mean(),
                                                 ((ratio - 1).abs() > c.clip).float().mean(),
                                                 vl.detach(), pl.detach()]))
            # One device sync per epoch; the KL-targeted learning rate adapts per epoch.
            e = torch.stack(ep_stats).cpu().numpy()
            ep_stats.clear()
            kls += list(e[:, 0]); clips += list(e[:, 1]); vls += list(e[:, 2]); pls += list(e[:, 3])
            kl = float(e[:, 0].mean())
            if c.lr_schedule == "adaptive":
                if kl > 2.0 * c.target_kl:
                    lr = max(1e-5, lr / 1.5)
                elif kl < 0.5 * c.target_kl:
                    lr = min(1e-2, lr * 1.5)
                for g in opt.param_groups:
                    g["lr"] = lr
        actor_cpu.load_state_dict({k: v.cpu() for k, v in actor.state_dict().items()})
        critic_cpu.load_state_dict({k: v.cpu() for k, v in critic.state_dict().items()})
        t_upd = time.time() - t_upd
        it += 1
        y = ret.reshape(B)
        ev = float(1.0 - np.var(y - buf["val"].reshape(B)) / (np.var(y) + 1e-8))

        if it % 5 == 0 or time.time() - start >= c.budget_s:
            done, fell, rsum, lsum, v2, r2, alive = win
            row = {"steps": steps, "time": round(time.time() - start, 1), "fps": int(steps / (time.time() - start)),
                   "scale": round(venv.scale, 3),
                   "fell_frac": fell / done if done else np.nan,
                   "ep_return": rsum / done if done else np.nan, "ep_len": lsum / done if done else np.nan,
                   "speed_rms": np.sqrt(v2 / max(alive, 1)), "rate_rms_dps": np.degrees(np.sqrt(r2 / max(alive, 1))),
                   "falls_per_robot_min": fell / max(alive, 1) * (60 / env_cfg.policy_dt),
                   "std": actor_cpu.log_std.exp().item(), "kl": np.mean(kls), "clip_frac": np.mean(clips),
                   "v_loss": np.mean(vls), "pg_loss": np.mean(pls), "ev": ev, "lr": lr,
                   "t_roll": round(t_roll, 3), "t_upd": round(t_upd, 3)}
            row = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in row.items()}
            win[:] = 0
            if writer is None:
                writer = csv.DictWriter(log, fieldnames=list(row))
                writer.writeheader()
            writer.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in row.items()})
            log.flush()
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        if steps >= next_ckpt:
            save(out / f"ckpt_{steps // 1_000_000:04d}M.pt", actor_cpu, onorm, c, env_cfg)
            next_ckpt += c.ckpt_every
    save(out / "final.pt", actor_cpu, onorm, c, env_cfg)
    venv.close()
    log.close()


def save(path, actor, onorm, c, env_cfg):
    torch.save({"actor": actor.state_dict(), "obs_norm": onorm.state(), "ppo": asdict(c),
                "env": asdict(env_cfg)}, path)


def load_policy(path):
    """Deterministic policy: obs (n, od) float32 -> command (n,)."""
    ck = torch.load(path, weights_only=False)
    od = len(ck["obs_norm"]["mean"])
    norm = RunningNorm(od)
    norm.mean, norm.var = np.array(ck["obs_norm"]["mean"]), np.array(ck["obs_norm"]["var"])
    if ck.get("kind") == "td3" and ck["td3"].get("actor_priv"):
        # Teacher: also needs the true state and parameters (env.last_priv); never runs on the robot.
        from td3 import TanhActor
        c = ck["td3"]
        pn = RunningNorm(len(ck["priv_norm"]["mean"]))
        pn.mean, pn.var = np.array(ck["priv_norm"]["mean"]), np.array(ck["priv_norm"]["var"])
        from td3 import PRIV_STATE
        psel = slice(PRIV_STATE, None) if c["actor_priv"] == "params" else slice(None)
        actor = TanhActor(od + len(pn.mean[psel]), tuple(c["actor_hidden"]), ACT[c["activation"]])
        actor.load_state_dict(ck["actor"])

        def teacher(obs, priv):
            with torch.no_grad():
                x = torch.from_numpy(np.concatenate([norm(obs), pn(priv)[:, psel]], 1))
                return np.clip(torch.tanh(actor.net(x)).squeeze(-1).numpy(), -1.0, 1.0)

        teacher.needs_priv = True
        return teacher, ck
    if ck.get("kind") == "td3":
        from td3 import TanhActor
        c = ck["td3"]
        actor = TanhActor(od, tuple(c["actor_hidden"]), ACT[c["activation"]])
        squash = torch.tanh
    else:
        c = ck["ppo"]
        if c.get("recurrent"):
            actor = RecActor(od, c["rnn_hidden"], tuple(c["actor_hidden"]), ACT[c["activation"]], c["log_std_init"])
            actor.load_state_dict(ck["actor"])
            state = {"h": None}

            def rpolicy(obs):
                x = torch.from_numpy(norm(obs))
                if state["h"] is None or state["h"].shape[0] != x.shape[0]:
                    state["h"] = torch.zeros(x.shape[0], actor.rnn_hidden)
                with torch.no_grad():
                    d, state["h"] = actor.step(x, state["h"], torch.zeros(x.shape[0]))
                return np.clip(d.mean.squeeze(-1).numpy(), -1.0, 1.0)

            return rpolicy, ck
        actor = Actor(od, tuple(c["actor_hidden"]), ACT[c["activation"]], c["log_std_init"])
        squash = lambda x: x
        if not c["norm_obs"]:
            norm.mean, norm.var = np.zeros(od), np.ones(od) - 1e-8
    actor.load_state_dict(ck["actor"])

    def policy(obs):
        with torch.no_grad():
            return np.clip(squash(actor.net(torch.from_numpy(norm(obs)))).squeeze(-1).numpy(), -1.0, 1.0)

    return policy, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--json", type=str, default="{}", help="PPOConfig overrides as JSON")
    args = ap.parse_args()
    over = json.loads(args.json)
    for f in fields(PPOConfig):
        if f.name in over and isinstance(getattr(PPOConfig, f.name, None), tuple):
            over[f.name] = tuple(over[f.name])
    c = PPOConfig(**over)
    train(c, args.out)


if __name__ == "__main__":
    main()
