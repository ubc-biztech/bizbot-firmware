"""Score every checkpoint of one or more runs, deterministic, on fixed robots.

    .venv/bin/python -u model/mj/evalckpt.py model/mj/runs/<run> [...] [--rows nominal "50% ranges"]

Writes eval.csv into each run and prints learning curves.
"""

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import metrics  # noqa: E402

ROWS = ("nominal", "50% ranges", "full ranges")
SKIP = ("n_envs", "policy_dt", "episode_s")


def one(job):
    path, row, n, seconds = job
    from ppo import load_policy
    policy, ck = load_policy(path)
    env = ck["env"]
    rand, scale = metrics.ROWS[row]
    if getattr(policy, "needs_priv", False):
        make = lambda e: (lambda m, obs: policy(obs, e.last_priv))
    else:
        make = lambda e: (lambda m, obs: policy(obs))
    r = metrics.run(None, rand, scale, n=n, seconds=seconds, make=make,
                    policy_dt=env["policy_dt"], env_kwargs={k: v for k, v in env.items() if k not in SKIP})
    return str(path), row, r


def ckpts(run):
    out = sorted(run.glob("ckpt_*M.pt"))
    if (run / "final.pt").exists():
        out.append(run / "final.pt")
    return out


def steps_of(path):
    return path.stem.split("_")[1] if path.stem.startswith("ckpt") else "final"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--rows", nargs="*", default=list(ROWS))
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()
    jobs = [(p, row, args.n, args.seconds) for run in args.runs for p in ckpts(run) for row in args.rows]
    res = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for path, row, r in ex.map(one, jobs):
            res[(path, row)] = r
    for run in args.runs:
        with open(run / "eval.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ckpt", "row", "fell", "speed", "rate", "hf", "dcmd", "sat", "creep"])
            print(f"\n== {run.name}\n{'ckpt':>8s} {'row':14s}{metrics.HEADER}")
            for p in ckpts(run):
                for row in args.rows:
                    r = res[(str(p), row)]
                    w.writerow([steps_of(p), row] + [f"{r[k]:.4g}" for k in ("fell", "speed", "rate", "hf", "dcmd", "sat", "creep")])
                    print(f"{steps_of(p):>8s} {row:14s}{metrics.fmt(r)}")


if __name__ == "__main__":
    main()
