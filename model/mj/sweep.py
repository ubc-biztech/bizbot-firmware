"""Run experiments one after another, each with all cores for a fixed time,
then score their checkpoints.

    .venv/bin/python -u model/mj/sweep.py --group r1 [--only name ...]

Each experiment is a PPOConfig override; see GROUPS.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PY = sys.executable

BASE = dict(workers=12, per_worker=256, budget_s=300, ckpt_every=5_000_000)

# The previous recipe (model/train.py, runs 2026-10-22*), as close as this trainer gets.
OLD = dict(asym=False, norm_obs=False, activation="tanh", actor_hidden=[64, 64], critic_hidden=[256, 256],
           lr=3e-4, lr_schedule="constant", gamma=0.995, lam=1.0, ent_coef=0.01, n_steps=64, minibatches=12,
           env_overrides=dict(reward_form="capped", history=40))
NEW = dict(asym=True, norm_obs=True, activation="elu", actor_hidden=[128, 128], critic_hidden=[256, 256],
           lr=3e-4, lr_schedule="adaptive", target_kl=0.01, gamma=0.99, lam=0.95, ent_coef=0.0,
           n_steps=32, minibatches=4, env_overrides=dict(reward_form="exp", history=16))

GROUPS = {
    "r1": {
        "old-recipe-nominal": {**OLD, "scale": 0.0},
        "new-full": {**NEW, "scale": 1.0},
        "new-half": {**NEW, "scale": 0.5},
        "new-curriculum": {**NEW, "scale": 0.0, "curriculum_s": 240},
        "new-symmetric-critic": {**NEW, "scale": 0.5, "asym": False},
        "new-capped-reward": {**NEW, "scale": 0.5, "env_overrides": dict(reward_form="capped", history=16)},
        "new-ent0.01": {**NEW, "scale": 0.5, "ent_coef": 0.01},
        "new-old-sim-nominal": {**NEW, "env": "old", "scale": 0.0, "env_overrides": dict(history=40)},
    },
}

TD3 = dict(algo="td3", workers=12, per_worker=256, batch=16384, updates_per_step=1, nstep=3,
           gamma=0.99, tau=0.05, noise_min=0.01, noise_max=0.2, asym=True,
           env_overrides=dict(reward_form="exp", history=16))
ALIVE = dict(reward_form="alive", shape_weight=0.5, history=16)
GROUPS["r2"] = {
    "ppo-half-alive": {**NEW, "scale": 0.5, "env_overrides": ALIVE},
    "ppo-full-alive": {**NEW, "scale": 1.0, "env_overrides": ALIVE},
    "ppo-full-alive-ent": {**NEW, "scale": 1.0, "ent_coef": 0.005, "env_overrides": ALIVE},
    "td3-half-alive": {**TD3, "scale": 0.5, "env_overrides": ALIVE},
    "td3-full-alive": {**TD3, "scale": 1.0, "env_overrides": ALIVE},
    "ppo-half-alive-hist40": {**NEW, "scale": 0.5, "env_overrides": {**ALIVE, "history": 40}},
    "ppo-half-alive-gru": {**NEW, "scale": 0.5, "recurrent": True, "rnn_hidden": 64,
                           "env_overrides": {**ALIVE, "history": 4}},
    "ppo-half-alive-wobble": {**NEW, "scale": 0.5, "env_overrides": {**ALIVE, "w_rate": 0.5, "w_dcmd": 20.0}},
    "ppo-half-alive-g995": {**NEW, "scale": 0.5, "gamma": 0.995, "env_overrides": ALIVE},
}


L1 = dict(reward_form="l1", history=16)
GROUPS["r2b"] = {
    "ppo-half-l1": {**NEW, "scale": 0.5, "env_overrides": L1},
    "ppo-full-l1": {**NEW, "scale": 1.0, "env_overrides": L1},
    "ppo-full-l1-ent": {**NEW, "scale": 1.0, "ent_coef": 0.005, "env_overrides": L1},
    "td3-half-l1": {**TD3, "scale": 0.5, "env_overrides": L1},
    "td3-full-l1": {**TD3, "scale": 1.0, "env_overrides": L1},
    "ppo-half-l1-hist40": {**NEW, "scale": 0.5, "env_overrides": {**L1, "history": 40}},
    "ppo-half-l1-gru": {**NEW, "scale": 0.5, "recurrent": True, "rnn_hidden": 64, "env_overrides": {**L1, "history": 4}},
    "ppo-half-exp-ent": {**NEW, "scale": 0.5, "ent_coef": 0.005},
}

EXP40 = dict(reward_form="exp", history=40)
L140 = dict(reward_form="l1", history=40)
GROUPS["r3"] = {
    "ppo-half-exp-ent-h40": {**NEW, "scale": 0.5, "ent_coef": 0.005, "env_overrides": EXP40},
    "ppo-full-exp-ent-h40": {**NEW, "scale": 1.0, "ent_coef": 0.005, "env_overrides": EXP40},
    "td3-full-exp-h40": {**TD3, "scale": 1.0, "env_overrides": EXP40},
    "td3-full-l1-h40": {**TD3, "scale": 1.0, "env_overrides": L140},
    "td3-full-l1-h40-v05": {**TD3, "scale": 1.0, "env_overrides": {**L140, "l1_v_scale": 0.5}},
    "td3-full-l1-h40-2upd": {**TD3, "scale": 1.0, "updates_per_step": 2, "env_overrides": L140},
    "ppo-half-exp-ent-h40-wobble": {**NEW, "scale": 0.5, "ent_coef": 0.005,
                                    "env_overrides": {**EXP40, "w_rate": 0.5, "w_dcmd": 20.0}},
}

# Seed repeats of the round-2b/3 leaders.
_LEAD = {
    "ppo-half-exp-ent": GROUPS["r2b"]["ppo-half-exp-ent"],
    "ppo-half-exp-ent-h40-wobble": GROUPS["r3"]["ppo-half-exp-ent-h40-wobble"],
    "td3-full-l1-h40-2upd": GROUPS["r3"]["td3-full-l1-h40-2upd"],
}
GROUPS["r4"] = {f"{k}-s{seed}": {**v, "seed": seed} for seed in (1, 2) for k, v in _LEAD.items()}

# TD3 wobble weight: base is r3 td3-full-l1-h40-2upd (l1_vel 0.6, l1_rate 0.2 over 2 rad/s).
_T = GROUPS["r3"]["td3-full-l1-h40-2upd"]
_W = {"w1": dict(l1_vel=0.5, l1_rate=0.3, l1_rate_scale=0.5), "w2": dict(l1_vel=0.4, l1_rate=0.4, l1_rate_scale=0.5)}
GROUPS["r5"] = {f"td3-{w}-s{seed}": {**_T, "seed": seed, "env_overrides": {**_T["env_overrides"], **o}}
                for w, o in _W.items() for seed in (0, 1, 2)}

# TD3 complexity: base is the r3 2-update config; reward set from r5 at launch time via R6_ENV.
R6_ENV = {**_T["env_overrides"], "l1_hold": 0.2}   # r7 winner
_C = {"redq5": dict(n_critics=5),
      "ln": dict(critic_layernorm=True),
      "ln-big": dict(critic_layernorm=True, critic_hidden=[1024, 512, 256]),
      "actor256": dict(actor_hidden=[256, 256])}
GROUPS["r6"] = {f"td3-{v}-s{seed}": {**_T, **o, "seed": seed, "env_overrides": R6_ENV}
                for v, o in _C.items() for seed in (0, 1)}

# TD3 drift: penalise the leaky wheel-rotation integral the policy observes (r3 2-update base).
GROUPS["r7"] = {f"td3-hold{w}-s{seed}": {**_T, "seed": seed, "env_overrides": {**_T["env_overrides"], "l1_hold": w}}
                for w in (0.2, 0.4) for seed in (0, 1, 2)}

# Teacher (actor sees true state and parameters) on the r7 winner's setup, to estimate headroom.
# Runs beside the overnight PBT run, so fewer workers and longer; compare at equal steps.
GROUPS["r8"] = {f"td3-teacher-s{seed}": {**_T, "seed": seed, "actor_priv": True, "env_overrides": R6_ENV,
                                         "workers": 6, "budget_s": 600}
                for seed in (0, 1, 2)}

# Teacher that sees only the robot's parameters, not its live true state: the realistic target
# for a student that infers the robot from sensor history.
GROUPS["r9"] = {f"td3-teacher-params-s{seed}": {**GROUPS["r8"][f"td3-teacher-s{seed}"], "actor_priv": "params"}
                for seed in (0, 1, 2)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", type=Path, default=HERE / "runs")
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()
    group = GROUPS[args.group]
    names = args.only or list(group)
    for name in names:
        cfg = {**BASE, **group[name]}
        script = "td3.py" if cfg.pop("algo", "ppo") == "td3" else "ppo.py"
        out = args.out / args.group / name
        out.mkdir(parents=True, exist_ok=True)
        t = time.time()
        print(f"=== {name}: {json.dumps(cfg)}", flush=True)
        with open(out / "train.log", "w") as log:
            subprocess.run([PY, "-u", str(HERE / script), "--out", str(out), "--json", json.dumps(cfg)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        lines = (out / "train.log").read_text().strip().splitlines()
        print(f"    trained {time.time() - t:.0f} s; last: {lines[-1] if lines else '-'}", flush=True)
        if not args.no_eval and cfg.get("env", "mj") == "mj":
            subprocess.run([PY, "-u", str(HERE / "evalckpt.py"), str(out)], cwd=ROOT, check=False)


if __name__ == "__main__":
    main()
