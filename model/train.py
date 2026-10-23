"""Train a balance policy with PPO (stable-baselines3) in the Python sim.

    .venv/bin/python model/train.py --steps 200000000 --out model/runs/first

By default the policy learns a correction added to the LQR's command
(--base none: the policy commands alone).

Robots are drawn from sim/bizbot_sim/params.py's ranges. Training starts with
the nominal robot only and widens the ranges (the env's `scale`) each time
few enough recent episodes end in a fall, and at least linearly to the full
ranges by half the run. Writes progress.csv, checkpoints,
config.json and final.zip to --out.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim"))
from bizbot_sim import EnvConfig, Randomization  # noqa: E402

from vec_env import BizBotVecEnv  # noqa: E402

HISTORY = 40          # policy steps of sensor history: 0.4 s at 100 Hz
SCALE_STEP = 0.05     # range widening per curriculum step
FELL_TARGET = 0.10    # widen when fewer than this share of recent episodes fell


class Curriculum(BaseCallback):
    def __init__(self, venv, total_steps):
        super().__init__()
        self.venv = venv
        self.total_steps = total_steps

    def _on_rollout_end(self):
        fell = self.venv.recent_fell
        rate = float(np.mean(fell)) if fell else 1.0
        if len(fell) >= 500 and rate < FELL_TARGET and self.venv.scale < 1.0:
            self.venv.scale = min(1.0, self.venv.scale + SCALE_STEP)
            fell.clear()
        floor = min(1.0, self.num_timesteps / (0.5 * self.total_steps))
        if floor > self.venv.scale:
            self.venv.scale = floor
        self.logger.record("curriculum/scale", self.venv.scale)
        self.logger.record("curriculum/fell_rate", rate)

    def _on_step(self):
        return True


class Progress(BaseCallback):
    """One tqdm line on the console; progress.csv has everything."""

    def __init__(self, total_steps):
        super().__init__()
        self.total_steps = total_steps

    def _on_training_start(self):
        self.bar = tqdm(total=self.total_steps, unit="step", unit_scale=True, smoothing=0.05)

    def _on_step(self):
        self.bar.update(self.training_env.num_envs)
        return True

    def _on_rollout_end(self):
        values, episodes = self.logger.name_to_value, self.model.ep_info_buffer
        self.bar.set_postfix(
            scale=f"{values.get('curriculum/scale', 0.0):.2f}",
            fell=f"{values.get('curriculum/fell_rate', 1.0):.2f}",
            ep_rew=f"{np.mean([e['r'] for e in episodes]):.0f}" if episodes else "-",
            std=f"{self.model.policy.log_std.exp().mean().item():.3f}",
            refresh=False)

    def _on_training_end(self):
        self.bar.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200_000_000)
    ap.add_argument("--n-envs", type=int, default=1024)
    ap.add_argument("--out", type=Path, default=Path("model/runs/latest"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--base", choices=["lqr", "none"], default="lqr")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = EnvConfig(n_envs=args.n_envs, history=HISTORY, lqr_base=args.base == "lqr")
    rand = Randomization()
    (args.out / "config.json").write_text(json.dumps(
        {"env": asdict(cfg), "ranges": repr(rand), "args": {k: str(v) for k, v in vars(args).items()}},
        indent=2))

    raw = BizBotVecEnv(cfg, rand, seed=args.seed, scale=0.0)
    venv = VecNormalize(raw, norm_obs=False, norm_reward=True, gamma=0.995)

    # gae_lambda 1 over 2.56 s rollouts: credit from real returns, not the
    # critic, whose errors otherwise steer the residual away from the LQR.
    model = PPO(
        "MlpPolicy", venv,
        n_steps=256, batch_size=16384, n_epochs=5, learning_rate=3e-4,
        gamma=0.995, gae_lambda=1.0, clip_range=0.2, ent_coef=0.01,
        vf_coef=0.5, max_grad_norm=1.0, seed=args.seed, device="cpu",
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[256, 256]),
                           activation_fn=torch.nn.Tanh, log_std_init=-1.0),
    )
    model.set_logger(configure(str(args.out), ["csv"]))
    callbacks = CallbackList([
        Curriculum(raw, args.steps),
        Progress(args.steps),
        CheckpointCallback(save_freq=max(1, 20_000_000 // args.n_envs), save_path=str(args.out / "checkpoints")),
    ])
    model.learn(total_timesteps=args.steps, callback=callbacks)
    model.save(args.out / "final")


if __name__ == "__main__":
    main()
