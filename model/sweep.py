"""Short residual-PPO tests, one process per variant, all at once.

    .venv/bin/python -u model/sweep.py

Each variant trains to the last of PROBE_AT and is scored at each of them on
nominal robots and on 50% ranges: reward per step and falls per robot per
20 s, deterministic, next to the LQR alone. Edit VARIANTS for the next test.
"""

import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "model"))

CREDIT = dict(gae_lambda=1.0, n_steps=256, gamma=0.995)
VARIANTS = {  # name: (EnvConfig overrides, randomization scale trained at, PPO overrides)
    "train at 0%": (dict(), 0.0, CREDIT),
    "train at 50%": (dict(), 0.5, CREDIT),
    "train at 100%": (dict(), 1.0, CREDIT),
}
PROBE_AT = (5_000_000, 10_000_000)


def score(policy, overrides, scale, n=256, seconds=20.0):
    from bizbot_sim import BizBotEnv, EnvConfig, Randomization
    cfg = EnvConfig(n_envs=n, history=40, lqr_base=True, **overrides)
    env = BizBotEnv(cfg, Randomization(), seed=5, scale=scale)
    obs, total, falls = env._obs(), 0.0, 0
    steps = int(round(seconds / cfg.policy_dt))
    for _ in range(steps):
        action = policy.predict(obs, deterministic=True)[0] if policy else np.zeros((n, 1))
        obs, reward, terminated, _, _ = env.step(action)
        total += float(reward.mean())
        falls += int(terminated.sum())
    return total / steps, falls / n


def run_variant(name):
    import torch
    from bizbot_sim import EnvConfig, Randomization
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import VecNormalize
    from vec_env import BizBotVecEnv

    class Report(BaseCallback):
        def _on_rollout_end(self):
            print(json.dumps({"progress": self.num_timesteps}), flush=True)

        def _on_step(self):
            return True

    torch.set_num_threads(2)
    overrides, scale, ppo = VARIANTS[name]
    ppo = {**dict(n_steps=64, batch_size=16384, n_epochs=5, learning_rate=3e-4,
                  gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                  vf_coef=0.5, max_grad_norm=1.0), **ppo}
    cfg = EnvConfig(n_envs=1024, history=40, lqr_base=True, **overrides)
    venv = VecNormalize(BizBotVecEnv(cfg, Randomization(), seed=0, scale=scale),
                        norm_obs=False, norm_reward=True, gamma=ppo["gamma"])
    model = PPO(
        "MlpPolicy", venv, seed=0, device="cpu",
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[256, 256]),
                           activation_fn=torch.nn.Tanh, log_std_init=-1.0),
        **ppo,
    )
    done = 0
    for at in PROBE_AT:
        model.learn(total_timesteps=at - done, reset_num_timesteps=False, callback=Report())
        done = at
        print(json.dumps({"name": name, "at": at, "nominal": score(model, overrides, 0.0),
                          "half": score(model, overrides, 0.5)}), flush=True)


def main():
    if len(sys.argv) > 1:
        run_variant(sys.argv[1])
        return
    start = time.time()
    procs, rows = {}, []
    for i, name in enumerate(VARIANTS):
        err = tempfile.TemporaryFile(mode="w+")
        p = subprocess.Popen([sys.executable, "-u", __file__, name], cwd=ROOT, text=True,
                             stdout=subprocess.PIPE, stderr=err)
        bar = tqdm(total=PROBE_AT[-1], desc=f"{name:16s}", position=i, unit="step", unit_scale=True)
        procs[name] = (p, err, bar)

    def read(p, bar):
        for line in p.stdout:
            msg = json.loads(line)
            if "progress" in msg:
                bar.update(msg["progress"] - bar.n)
            else:
                rows.append(msg)

    readers = [threading.Thread(target=read, args=(p, bar)) for p, _, bar in procs.values()]
    for t in readers:
        t.start()
    base = {"nominal": score(None, {}, 0.0), "half": score(None, {}, 0.5)}
    for t in readers:
        t.join()
    errors = {}
    for name, (p, err, bar) in procs.items():
        p.wait()
        bar.close()
        if p.returncode != 0:
            err.seek(0)
            lines = err.read().strip().splitlines()
            errors[name] = lines[-1] if lines else f"exit {p.returncode}"

    order = list(VARIANTS)
    rows.sort(key=lambda r: (order.index(r["name"]), r["at"]))
    cell = lambda v: f"{v[0]:+8.3f} {v[1]:6.2f}"
    print(f"\n{'':36s}{'nominal robots':>16s}{'50% ranges':>16s}")
    print(f"{'':36s}{'reward  falls':>16s}{'reward  falls':>16s}")
    print(f"{'LQR alone':36s}{cell(base['nominal']):>16s}{cell(base['half']):>16s}")
    for r in rows:
        label = f"{r['name']} @ {r['at'] / 1e6:.0f}M"
        print(f"{label:36s}{cell(r['nominal']):>16s}{cell(r['half']):>16s}")
    for name, e in errors.items():
        print(f"{name}: FAILED: {e}")
    print(f"({time.time() - start:.0f} s; reward per step, falls per robot per 20 s)")


if __name__ == "__main__":
    main()
