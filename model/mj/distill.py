"""Distil a privileged teacher into a sensor-only student (DAgger).

    .venv/bin/python -u model/mj/distill.py --teacher model/mj/runs/teacher1/final.pt --out model/mj/runs/student1

The teacher sees the ESP32's observation plus privileged values (env.last_priv);
the student sees only the observation, through the same network shape as a TD3
actor. Every step, each robot is driven by the teacher with probability beta
(falling linearly from 1 to 0 over beta_steps samples) and otherwise by the
student; the teacher labels every visited state with its own command, and the
student is regressed onto those labels from a replay of recent states. The
student uses the teacher's observation normalisation and is saved as a TD3
checkpoint (ckpt_*M.pt, final.pt), so ppo.load_policy, evalckpt and watch.py
read it unchanged.
"""

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(HERE))
from ppo import ACT, RunningNorm, load_policy  # noqa: E402
from td3 import TanhActor, TD3Config, save  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--per-worker", type=int, default=256)
    ap.add_argument("--budget-s", type=float, default=3600)
    ap.add_argument("--beta-steps", type=float, default=20e6)
    ap.add_argument("--buffer", type=int, default=2_000_000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--updates-per-step", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--noise", type=float, default=0.02, help="exploration noise on executed commands")
    ap.add_argument("--ckpt-every", type=float, default=10e6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--student-history", type=int, default=0,
                    help="frames of history the student sees (default: the teacher's); the teacher always gets its own")
    args = ap.parse_args()

    from bizbot_mj import Randomization
    from bizbot_mj.env import EnvConfig
    from vec import VecEnv
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    teacher, tck = load_policy(args.teacher)
    assert getattr(teacher, "needs_priv", False), "the teacher must be a privileged TD3 checkpoint"
    tc = tck["td3"]
    env_kwargs = {k: v for k, v in tck["env"].items() if k != "n_envs"}
    th = env_kwargs["history"]
    sh = args.student_history or th
    assert sh >= th, "the student's history must contain the teacher's"
    env_kwargs["history"] = sh
    env_cfg = EnvConfig(n_envs=args.per_worker, **env_kwargs)
    FR = 4   # BalanceEnv.FRAME: values per history frame; the observation ends with 2 extras

    def teacher_obs(o):
        """The teacher's own observation: its last th frames and the extras."""
        return np.concatenate([o[:, (sh - th) * FR:sh * FR], o[:, -2:]], 1) if sh != th else o
    venv = VecEnv(env_cfg, Randomization(), args.workers, seed=args.seed, scale=1.0)
    N, od = venv.n, venv.obs_size

    # The student's normaliser: the teacher's, with its newest frame's statistics repeated for older frames.
    tm, tv = np.array(tck["obs_norm"]["mean"]), np.array(tck["obs_norm"]["var"])
    if sh != th:
        frame_m, frame_v = tm[(th - 1) * FR:th * FR], tv[(th - 1) * FR:th * FR]
        tm = np.concatenate([np.tile(frame_m, sh - th), tm])
        tv = np.concatenate([np.tile(frame_v, sh - th), tv])
    onorm = RunningNorm(od)
    onorm.mean, onorm.var = tm, tv
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    act = ACT[tc["activation"]]
    student = TanhActor(od, tuple(tc["actor_hidden"]), act).to(dev)
    student_cpu = TanhActor(od, tuple(tc["actor_hidden"]), act)
    student_cpu.load_state_dict(student.state_dict())
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    scfg = replace(TD3Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in tc.items()
                                if k in TD3Config.__dataclass_fields__}), actor_priv=False)
    (args.out / "config.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                                                       "teacher_td3": tc, "env": env_kwargs}, indent=1))

    S = args.buffer
    B_o = torch.zeros(S, od, device=dev)
    B_a = torch.zeros(S, device=dev)
    ptr = size = 0
    obs, priv = venv.observe()
    log = open(args.out / "progress.csv", "w", newline="")
    w = csv.writer(log)
    w.writerow(["samples", "time", "fps", "beta", "loss", "fell_frac", "falls_per_robot_min"])
    start = time.time()
    samples, steps, next_ckpt = 0, 0, args.ckpt_every
    losses, win = [], np.zeros(7)
    while time.time() - start < args.budget_s:
        o_n = onorm(obs)
        a_t = teacher(teacher_obs(obs), priv).astype(np.float32)
        with torch.no_grad():
            a_s = student_cpu(torch.from_numpy(o_n)).squeeze(-1).numpy()
        beta = max(0.0, 1.0 - samples / args.beta_steps)
        use_t = rng.random(N) < beta
        a = np.where(use_t, a_t, a_s) + args.noise * rng.standard_normal(N)
        idx = (ptr + np.arange(N)) % S
        ti = torch.from_numpy(idx).to(dev)
        B_o[ti] = torch.from_numpy(o_n).to(dev)
        B_a[ti] = torch.from_numpy(a_t).to(dev)
        ptr, size = (ptr + N) % S, min(S, size + N)
        obs, priv, _, _, _, _, _, st = venv.step(np.clip(a, -1.0, 1.0).astype(np.float32))
        win += st
        samples += N
        steps += 1
        for _ in range(args.updates_per_step):
            b = torch.randint(0, size, (args.batch,), device=dev)
            loss = ((student(B_o[b]).squeeze(-1) - B_a[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.detach())
        if steps % 5 == 0:
            student_cpu.load_state_dict({k: v.cpu() for k, v in student.state_dict().items()})
        if steps % 500 == 0:
            done, fell, _, _, _, _, alive = win
            row = [samples, round(time.time() - start, 1), int(samples / (time.time() - start)), round(beta, 3),
                   float(torch.stack(losses).mean()), fell / done if done else float("nan"),
                   fell / max(alive, 1) * (60 / env_cfg.policy_dt)]
            w.writerow(row)
            log.flush()
            print(json.dumps(dict(zip(["samples", "time", "fps", "beta", "loss", "fell_frac", "falls_per_robot_min"],
                                      [round(x, 5) if isinstance(x, float) else x for x in row]))), flush=True)
            losses, win = [], np.zeros(7)
        if samples >= next_ckpt:
            student_cpu.load_state_dict({k: v.cpu() for k, v in student.state_dict().items()})
            save(args.out / f"ckpt_{samples // 1_000_000:04d}M.pt", student_cpu, onorm, scfg, env_cfg)
            next_ckpt += args.ckpt_every
    student_cpu.load_state_dict({k: v.cpu() for k, v in student.state_dict().items()})
    save(args.out / "final.pt", student_cpu, onorm, scfg, env_cfg)
    venv.close()
    log.close()


if __name__ == "__main__":
    main()
