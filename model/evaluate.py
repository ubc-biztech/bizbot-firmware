"""Survival table: controllers side by side on the same simulated robots.

    .venv/bin/python model/evaluate.py [--policy model/runs/latest/final.zip]

Each cell is the share of robots still up after 20 s, standing still with the
robot's own random pushes. Rows are the randomization width and an IMU latency
sweep on the nominal robot, including latencies outside the training ranges.
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim"))
from bizbot_sim import BizBotEnv, EnvConfig, Randomization, U  # noqa: E402
from bizbot_sim.controllers import CONTROL_DT, cascaded_pid, firmware_pid, lqr  # noqa: E402

# kp from the team's history; ki is the smallest that stands in the sim.
HISTORY_PID = dict(kp=0.04, ki=0.2, kd=0.001, cap=0.2, alpha=0.3)

ROWS = [("nominal robot", Randomization(), 0.0), ("25% ranges", Randomization(), 0.25),
        ("50% ranges", Randomization(), 0.5), ("full ranges", Randomization(), 1.0)]
ROWS += [(f"IMU latency {ms} ms", replace(Randomization(), imu_latency=U(ms / 1000, ms / 1000, ms / 1000)), 0.0)
         for ms in (0, 20, 40, 60, 80, 100, 120)]


def survival(rand, scale, n, seconds, seed, ctrl_factory=None, policy=None, env_cfg=None):
    if policy is None:
        cfg = EnvConfig(n_envs=n, policy_dt=CONTROL_DT, action_rate_limit=2.0, episode_s=seconds + 1)
    else:
        cfg = EnvConfig(**{**env_cfg, "n_envs": n, "episode_s": seconds + 1})
    env = BizBotEnv(cfg, rand, seed=seed, scale=scale)
    env.linear[:] = 0.0
    env.vref_next[:] = np.inf
    obs = env._obs()
    ctrl = ctrl_factory(n) if policy is None else None
    fell = np.zeros(n, dtype=bool)
    for _ in range(int(round(seconds / cfg.policy_dt))):
        if policy is None:
            action = ctrl(env.plant.measurements())
        else:
            action, _ = policy.predict(obs, deterministic=True)
        obs, _, terminated, _, _ = env.step(action)
        fell |= terminated
    return 1.0 - fell.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    columns = [("firmware PID", lambda n: firmware_pid(n, **HISTORY_PID), None),
               ("HTML PID", cascaded_pid, None), ("LQR", lqr, None)]
    env_cfg = None
    if args.policy:
        from stable_baselines3 import PPO
        env_cfg = json.loads((args.policy.parent / "config.json").read_text())["env"]
        name = "LQR + RL" if env_cfg.get("lqr_base") else "RL policy"
        columns.append((name, None, PPO.load(args.policy, device="cpu")))

    print(f"{'':22s}" + "".join(f"{name:>14s}" for name, _, _ in columns))
    for label, rand, scale in ROWS:
        cells = [survival(rand, scale, args.n, args.seconds, args.seed, f, p, env_cfg) for _, f, p in columns]
        print(f"{label:22s}" + "".join(f"{c * 100:13.0f}%" for c in cells), flush=True)


if __name__ == "__main__":
    main()
