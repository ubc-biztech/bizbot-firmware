"""Score each PBT generation's best member on the fixed baseline robots.

    .venv/bin/python model/mj/pbt_track.py [model/mj/runs/pbt1]

The population scores itself on new robots each generation, which makes
generations hard to compare with each other or with the LQR baselines. This
re-scores each generation's top-ranked member on the same robots as
baselines.py and evalckpt.py (metrics.run's default seed), and writes them to
<run>/track/eval.csv and progress.csv as one learning curve (checkpoints named
by that member's samples). Generations already scored are skipped.
"""

import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evalckpt import ROWS, one  # noqa: E402

COLS = ("fell", "speed", "rate", "hf", "dcmd", "sat", "creep")


def main():
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "runs" / "pbt1"
    track = run / "track"
    track.mkdir(exist_ok=True)
    best = {}
    for r in csv.DictReader(open(run / "pbt.csv")):
        if r["rank"] == "0":
            best[int(r["gen"])] = (int(r["member"]), int(r["samples"]))
    ev_path = track / "eval.csv"
    done = set()
    rows = []
    if ev_path.exists():
        rows = list(csv.DictReader(open(ev_path)))
        done = {int(r["gen"]) for r in rows}
    todo = [(g, m, s) for g, (m, s) in sorted(best.items()) if g not in done]
    jobs = [(run / f"gen_{g:03d}" / f"m{m}.pt", row, 128, 20.0) for g, m, s in todo for row in ROWS]
    res = {}
    with ProcessPoolExecutor(max_workers=3) as ex:
        for path, row, r in ex.map(one, jobs):
            res[(path, row)] = r
    for g, m, s in todo:
        for row in ROWS:
            r = res[(str(run / f"gen_{g:03d}" / f"m{m}.pt"), row)]
            rows.append({"gen": g, "member": m, "ckpt": f"{s // 1_000_000:04d}M", "row": row,
                         **{k: f"{r[k]:.4g}" for k in COLS}})
    rows.sort(key=lambda r: (int(r["gen"]), ROWS.index(r["row"])))
    with open(ev_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["gen", "member", "ckpt", "row", *COLS])
        w.writeheader()
        w.writerows(rows)
    if best:
        last = max(best)
        with open(track / "progress.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["steps"])
            w.writerow([best[last][1]])
    for g, m, s in todo:
        cells = " | ".join(f"{row}: falls {res[(str(run / f'gen_{g:03d}' / f'm{m}.pt'), row)]['fell']:.1%}" for row in ROWS)
        print(f"gen {g} member {m}: {cells}")


if __name__ == "__main__":
    main()
