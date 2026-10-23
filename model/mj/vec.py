"""Many BalanceEnvs in worker processes, exchanging arrays through shared memory.

Each worker owns `per_worker` robots in one MuJoCo model. step() writes the
actions, wakes every worker, and waits for all of them.
"""

import multiprocessing as mp
import sys
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STATS = ("done", "fell", "ret", "len", "v2", "r2", "alive")


def _arrays(spec, shm):
    out, off = {}, 0
    for name, shape, dtype in spec:
        n = int(np.prod(shape)) * np.dtype(dtype).itemsize
        out[name] = np.ndarray(shape, dtype=dtype, buffer=shm.buf, offset=off)
        off += n
    return out


def _spec(n, obs, priv):
    return [("act", (n,), np.float32), ("obs", (n, obs), np.float32), ("priv", (n, priv), np.float32),
            ("rew", (n,), np.float32), ("term", (n,), np.bool_), ("trunc", (n,), np.bool_),
            ("tobs", (n, obs), np.float32), ("tpriv", (n, priv), np.float32),
            ("stats", (n, len(STATS)), np.float64)]


def _size(spec):
    return sum(int(np.prod(s)) * np.dtype(d).itemsize for _, s, d in spec)


def _worker(conn, shm_name, spec, lo, hi, env_kind, cfg, rand, seed, scale):
    sys.path.insert(0, str(ROOT / "sim"))
    sys.path.insert(0, str(ROOT / "model" / "mj"))
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    shm = shared_memory.SharedMemory(name=shm_name)
    a = _arrays(spec, shm)
    sl = slice(lo, hi)
    if env_kind == "mj":
        from bizbot_mj.env import BalanceEnv
        env = BalanceEnv(cfg, rand, seed=seed, scale=scale)
    else:
        from oldenv import OldEnv
        env = OldEnv(cfg, rand, seed=seed, scale=scale)
    a["obs"][sl] = env.reset()
    a["priv"][sl] = env.last_priv
    conn.send("ready")
    while True:
        msg = conn.recv()
        if msg == "step":
            obs, rew, term, trunc, info = env.step(a["act"][sl])
            a["obs"][sl], a["priv"][sl] = obs, env.last_priv
            a["rew"][sl], a["term"][sl], a["trunc"][sl] = rew, term, trunc
            st = np.zeros((hi - lo, len(STATS)))
            if info:
                d = info["done"]
                a["tobs"][lo + d] = info["terminal_obs"]
                a["tpriv"][lo + d] = info["terminal_priv"]
                st[d, 0] = 1
                st[d, 1] = info["fell"]
                st[d, 2] = info["episode_return"]
                st[d, 3] = info["episode_length"]
            t = env.plant.truth()
            alive = ~term
            st[:, 4] = np.where(alive, t["speed"] ** 2, 0)
            st[:, 5] = np.where(alive, t["pitch_rate"] ** 2, 0)
            st[:, 6] = alive
            a["stats"][sl] = st
            conn.send(None)
        elif msg[0] == "scale":
            env.scale = msg[1]
            conn.send(None)
        elif msg == "close":
            break
    shm.close()


class VecEnv:
    def __init__(self, cfg, rand, workers, seed=0, scale=1.0, env_kind="mj"):
        from dataclasses import replace
        self.workers = workers
        self.per = cfg.n_envs
        self.n = workers * self.per
        sys.path.insert(0, str(ROOT / "sim"))
        if env_kind == "mj":
            from bizbot_mj.env import BalanceEnv
            probe = BalanceEnv(replace(cfg, n_envs=1), rand, seed=0, scale=0.0)
        else:
            from oldenv import OldEnv
            probe = OldEnv(replace(cfg, n_envs=1), rand, seed=0, scale=0.0)
        self.obs_size, self.priv_size = probe.obs_size, probe.priv_size
        del probe
        self.spec = _spec(self.n, self.obs_size, self.priv_size)
        self.shm = shared_memory.SharedMemory(create=True, size=_size(self.spec))
        self.a = _arrays(self.spec, self.shm)
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for w in range(workers):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, daemon=True,
                            args=(child, self.shm.name, self.spec, w * self.per, (w + 1) * self.per,
                                  env_kind, cfg, rand, seed * 1000 + w, scale))
            p.start()
            self.conns.append(parent)
            self.procs.append(p)
        for c in self.conns:
            assert c.recv() == "ready"
        self._scale = scale

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = float(value)
        for c in self.conns:
            c.send(("scale", self._scale))
        for c in self.conns:
            c.recv()

    def observe(self):
        return self.a["obs"].copy(), self.a["priv"].copy()

    def step(self, actions):
        self.a["act"][:] = actions
        for c in self.conns:
            c.send("step")
        for c in self.conns:
            c.recv()
        a = self.a
        return (a["obs"].copy(), a["priv"].copy(), a["rew"].copy(), a["term"].copy(), a["trunc"].copy(),
                a["tobs"], a["tpriv"], a["stats"].sum(axis=0))

    def close(self):
        for c in self.conns:
            try:
                c.send("close")
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
        self.shm.close()
        self.shm.unlink()
