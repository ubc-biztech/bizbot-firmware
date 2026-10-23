"""Build the JSON files behind the training dashboard from the runs on disk.

    .venv/bin/python model/mj/dashboard_data.py --out <dir> --status "..." [--baselines <json>]

Writes leaderboard.json (one row per configuration: mean over seeds of the
final checkpoint), curves.json (every checkpoint of every run in those
configurations), pbt.json (pbt.csv rows, if a PBT run exists) and status.json.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
ROWS = {"nominal": "nom", "50% ranges": "half", "full ranges": "full"}

# Label: (kind, trained on, run directories by glob)
CONFIGS = {
    "Overnight PBT, best member": ("TD3", "full", ["pbt1/track"]),
    "Teacher, knows true state (not deployable)": ("Teacher", "full", ["r8/td3-teacher-s*"]),
    "Teacher, knows parameters (not deployable)": ("Teacher", "full", ["r9/td3-teacher-params-s*", "teacher1"]),
    "Student, distilled from teacher": ("TD3", "full", ["student1"]),
    "Student, distilled + RL fine-tune": ("TD3", "full", ["student1_ft"]),
    "Student, distilled, 1 s history": ("TD3", "full", ["student_h100"]),
    "Student, distilled, 2 s history": ("TD3", "full", ["student_h200"]),
    "PPO, exploration bonus": ("PPO", "50%", ["r2b/ppo-half-exp-ent", "r4/ppo-half-exp-ent-s*"]),
    "PPO, wobble penalty": ("PPO", "50%", ["r3/ppo-half-exp-ent-h40-wobble", "r4/ppo-half-exp-ent-h40-wobble-s*"]),
    "TD3, 2 updates/step": ("TD3", "full", ["r3/td3-full-l1-h40-2upd", "r4/td3-full-l1-h40-2upd-s*"]),
    "TD3, wobble weight w1": ("TD3", "full", ["r5/td3-w1-s*"]),
    "TD3, wobble weight w2": ("TD3", "full", ["r5/td3-w2-s*"]),
    "TD3, distance penalty 0.2": ("TD3", "full", ["r7/td3-hold0.2-s*"]),
    "TD3, distance penalty 0.4": ("TD3", "full", ["r7/td3-hold0.4-s*"]),
    "TD3, 5-critic ensemble": ("TD3", "full", ["r6/td3-redq5-s*"]),
    "TD3, LayerNorm critic": ("TD3", "full", ["r6/td3-ln-s*"]),
    "TD3, big LayerNorm critic": ("TD3", "full", ["r6/td3-ln-big-s*"]),
    "TD3, 256x256 actor": ("TD3", "full", ["r6/td3-actor256-s*"]),
}


def runs_of(globs):
    out = []
    for g in globs:
        out += sorted(p for p in RUNS.glob(g) if (p / "eval.csv").exists())
    return out


def final_steps(run):
    rows = list(csv.DictReader(open(run / "progress.csv")))
    if not rows:
        return np.nan
    return float(rows[-1].get("steps") or rows[-1]["samples"]) / 1e6   # distillation logs samples


def num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--status", default="")
    ap.add_argument("--baselines", type=Path)
    ap.add_argument("--pbt", type=Path, default=RUNS / "pbt1")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    board, curves = [], []
    if args.baselines and args.baselines.exists():
        b = json.loads(args.baselines.read_text())
        for name, label in (("tuned LQR", "LQR, tuned per robot"), ("untuned LQR", "LQR, untuned")):
            row = {"config": label, "kind": "LQR", "trained_on": "-", "seeds": 1, "steps_m": None}
            for r, short in ROWS.items():
                v = b[f"{name}|{r}"]
                row[f"{short}_fell"], row[f"{short}_rate"], row[f"{short}_speed"] = num(v["fell"]), num(v["rate"]), num(v["speed"])
            board.append(row)

    for label, (kind, on, globs) in CONFIGS.items():
        runs = runs_of(globs)
        if not runs:
            continue
        finals = []
        for seed, run in enumerate(runs):
            ev = list(csv.DictReader(open(run / "eval.csv")))
            fsteps = final_steps(run)
            if "gen" in (ev[0] if ev else {}):
                last = max(e["ckpt"] for e in ev)
                ev = [dict(e, ckpt="final") if e["ckpt"] == last else e for e in ev]
            for e in ev:
                steps = fsteps if e["ckpt"] == "final" else float(e["ckpt"].rstrip("M"))
                curves.append({"config": label, "seed": seed, "run": f"{label} #{seed + 1}", "steps_m": round(steps, 2),
                               "range": e["row"], "fell": num(e["fell"]), "rate": num(e["rate"]), "speed": num(e["speed"])})
                if e["ckpt"] == "final":
                    finals.append((e["row"], num(e["fell"]), num(e["rate"]), num(e["speed"])))
        row = {"config": label, "kind": kind, "trained_on": on, "seeds": len(runs),
               "steps_m": round(float(np.mean([final_steps(r) for r in runs])), 1)}
        for r, short in ROWS.items():
            vals = [f for f in finals if f[0] == r]
            for i, m in ((1, "fell"), (2, "rate"), (3, "speed")):
                xs = [v[i] for v in vals if v[i] is not None]
                row[f"{short}_{m}"] = round(float(np.mean(xs)), 4) if xs else None
        board.append(row)

    pbt = []
    if (args.pbt / "pbt.csv").exists():
        for r in csv.DictReader(open(args.pbt / "pbt.csv")):
            pbt.append({"gen": int(r["gen"]), "time_h": num(r["time_h"]), "member": int(r["member"]),
                        "samples_m": round(float(r["samples"]) / 1e6, 2), "cost": num(r["cost"]), "rank": int(r["rank"]),
                        "action": r["action"], "soft_reset": int(r["soft_reset"]),
                        **{f"{ROWS[k.split('|')[0]]}_{k.split('|')[1]}": num(v) for k, v in r.items() if "|" in k}})

    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (args.out / "leaderboard.json").write_text(json.dumps(board))
    (args.out / "curves.json").write_text(json.dumps(curves))
    (args.out / "pbt.json").write_text(json.dumps(pbt))
    (args.out / "status.json").write_text(json.dumps([{"updated": now, "status": args.status}]))
    print(f"leaderboard {len(board)} rows, curves {len(curves)} rows, pbt {len(pbt)} rows")


if __name__ == "__main__":
    main()
