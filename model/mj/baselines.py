"""Classical controllers on the MuJoCo sim, every row in its own process.

    .venv/bin/python -u model/mj/baselines.py [--n 128] [--rows nominal "full ranges"]
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics  # noqa: E402

CONTROLLERS = ("untuned LQR", "tuned LQR", "firmware PID", "cascaded PID")


def one(args):
    ctrl, row, n, seconds = args
    rand, scale = metrics.ROWS[row]
    r = metrics.run(None, rand, scale, n=n, seconds=seconds,
                    make=lambda env: metrics.classical(ctrl, n, env))
    return ctrl, row, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--rows", nargs="*", default=list(metrics.ROWS))
    ap.add_argument("--controllers", nargs="*", default=list(CONTROLLERS))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    jobs = [(c, r, args.n, args.seconds) for r in args.rows for c in args.controllers]
    results = {}
    with ProcessPoolExecutor(max_workers=14) as ex:
        for ctrl, row, r in ex.map(one, jobs):
            results[(ctrl, row)] = r
            print(f"{row:24s} {ctrl:14s} {metrics.fmt(r)}", flush=True)
    print(f"\n{'':24s} {'':14s}{metrics.HEADER}")
    for row in args.rows:
        for c in args.controllers:
            print(f"{row:24s} {c:14s} {metrics.fmt(results[(c, row)])}")
    if args.out:
        args.out.write_text(json.dumps({f"{c}|{r}": v for (c, r), v in results.items()}, indent=1))


if __name__ == "__main__":
    main()
