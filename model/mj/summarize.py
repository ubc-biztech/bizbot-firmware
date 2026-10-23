"""Learning curves of several runs side by side, from their eval.csv files.

    .venv/bin/python model/mj/summarize.py model/mj/runs/r1/* [--row "50% ranges"] [--metric fell]
"""

import argparse
import csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--rows", nargs="*", default=["nominal", "50% ranges", "full ranges"])
    ap.add_argument("--metrics", nargs="*", default=["fell", "speed", "rate", "hf"])
    args = ap.parse_args()
    data = {}
    cols = set()
    for run in args.runs:
        f = run / "eval.csv"
        if not f.exists():
            continue
        for r in csv.DictReader(open(f)):
            data[(run.name, r["ckpt"], r["row"])] = r
            cols.add(r["ckpt"])
    cols = sorted(c for c in cols if c != "final") + ["final"]
    for row in args.rows:
        for m in args.metrics:
            print(f"\n[{row}] {m}" + ("  (share fallen)" if m == "fell" else ""))
            print(f"{'run':28s}" + "".join(f"{c:>8s}" for c in cols))
            for run in args.runs:
                cells = []
                for c in cols:
                    r = data.get((run.name, c, row))
                    cells.append(f"{float(r[m]):8.3f}" if r else f"{'':8s}")
                if any(x.strip() for x in cells):
                    print(f"{run.name:28s}" + "".join(cells))


if __name__ == "__main__":
    main()
