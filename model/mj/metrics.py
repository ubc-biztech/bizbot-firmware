"""Score controllers on the MuJoCo sim, all on the same robots.

Every controller sees only what the ESP32 sees. Robots stand still for 20 s
with their own random pushes and condition changes. Quality is measured from
2 s on, only while the robot is up:

  fell      share of robots that passed 35 degrees
  speed     RMS horizontal speed, m/s            (standing still: lower is better)
  creep     mean over robots of |average signed speed|, m/s (steady walking off)
  rate      RMS pitch rate, deg/s                (wobble)
  hf        RMS pitch rate above 2 Hz, deg/s     (fast wobble)
  dcmd      RMS command change per 10 ms         (motor chatter)
  sat       share of time the command is at +-1
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sim"))
from bizbot_mj import Randomization, fixed  # noqa: E402
from bizbot_mj.env import BalanceEnv, EnvConfig  # noqa: E402

CONTROL_DT = 0.005
WARMUP_S = 2.0


def _hp_rand(pl, i):
    """The robot's true values as an old-sim Randomization, for designing an LQR on it."""
    import bizbot_sim.params as hp
    p = pl.p
    Z = lambda v: hp.U(float(v), float(v), float(v))
    vals = dict(
        body_mass=p["body_mass"][i], com_height=p["com_height"][i], com_offset=p["com_offset"][i],
        body_gyration=p["gyration_real"][i], wheel_mass=p["wheel_mass"][i],
        wheel_inertia=p["wheel_inertia"][i], wheel_radius=p["wheel_radius"][i],
        back_emf=p["k_m"][:, i].mean(), phase_resistance=p["r_m"][:, i].mean(),
        current_limit=p["imax_m"][:, i].mean(), speed_kp=p["speed_kp"][i],
        speed_full_scale_rpm=p["speed_full_scale_rpm"][i], speed_ki_ratio=p["speed_ki_ratio"][i],
        viscous_friction=p["viscous_friction"][i], cmd_lowpass=p["cmd_lowpass"][i],
        fb_lag=p["fb_lag"][i], battery_ocv=p["battery_ocv"][i],
        imu_noise_deg=max(p["imu_noise_deg"][i], 0.01),
    )
    return replace(hp.Randomization(), **{k: Z(v) for k, v in vals.items()})


class OracleLQR:
    """LQR + Kalman filter designed per robot from its true parameters and IMU
    delay: the best case of perfect system identification ("tuned LQR")."""

    def __init__(self, env):
        from bizbot_sim.controllers import lqr
        self.env, self.lqr = env, lqr
        self.ctrls = [None] * env.cfg.n_envs
        self.design(np.arange(env.cfg.n_envs))

    def design(self, idx):
        pl = self.env.plant
        for i in idx:
            delay = pl.p["imu_latency"][i] + pl.p["imu_filter"][i] + 0.5 * pl.p["imu_period"][i]
            self.ctrls[i] = self.lqr(1, rand=_hp_rand(pl, i), imu_latency=float(delay),
                                     imu_noise_deg=max(float(pl.p["imu_noise_deg"][i]), 0.02))

    def __call__(self, m, obs):
        sl = lambda i: {k: v[i:i + 1] for k, v in m.items()}
        return np.concatenate([c(sl(i)) for i, c in enumerate(self.ctrls)])


def classical(name, n, env):
    from bizbot_sim.controllers import cascaded_pid, firmware_pid, lqr
    if name == "untuned LQR":
        c = lqr(n)
    elif name == "tuned LQR":
        return OracleLQR(env)
    elif name == "firmware PID":
        c = firmware_pid(n, kp=0.04, ki=0.2, kd=0.001, cap=0.2, alpha=0.3)
    elif name == "cascaded PID":
        c = cascaded_pid(n)
    elif name == "none":
        return lambda m, obs: np.zeros(n)
    else:
        raise KeyError(name)
    return lambda m, obs: c(m)


def run(controller, rand, scale, n=128, seconds=20.0, seed=12345, policy_dt=CONTROL_DT, env_kwargs=None,
        make=None):
    """controller(measurements, obs) -> command in [-1, 1], or make(env) builds it."""
    cfg = EnvConfig(n_envs=n, policy_dt=policy_dt, episode_s=seconds + 10.0,
                    action_rate_limit=2.0 if policy_dt == CONTROL_DT else (env_kwargs or {}).get("action_rate_limit", 0.25),
                    **{k: v for k, v in (env_kwargs or {}).items() if k != "action_rate_limit"})
    env = BalanceEnv(cfg, rand, seed=seed, scale=scale)
    if make is not None:
        controller = make(env)
    obs = env._obs()
    fell = np.zeros(n, dtype=bool)
    steps = int(round(seconds / cfg.policy_dt))
    warm = int(round(WARMUP_S / cfg.policy_dt))
    acc = {k: np.zeros(n) for k in ("v1", "v2", "r2", "hf2", "du2", "sat", "cnt")}
    rate_hist = []
    u_prev = env.u.copy()
    for k in range(steps):
        a = controller(env.plant.measurements(), obs)
        obs, _, term, _, _ = env.step(a)
        fell |= term
        t = env.plant.truth()
        rate_hist.append(np.where(fell, 0.0, t["pitch_rate"]))
        if k >= warm:
            up = ~fell
            acc["v1"] += np.where(up, t["speed"], 0)
            acc["v2"] += np.where(up, t["speed"] ** 2, 0)
            acc["r2"] += np.where(up, t["pitch_rate"] ** 2, 0)
            du = (env.u - u_prev) * (0.01 / cfg.policy_dt) ** 0.5
            acc["du2"] += np.where(up, du ** 2, 0)
            acc["sat"] += np.where(up, np.abs(env.u) > 0.98, 0)
            acc["cnt"] += up
        u_prev = env.u.copy()
    # Fast wobble: pitch rate above 2 Hz, after warm-up, robots that stayed up.
    rh = np.array(rate_hist)[warm:, ~fell]
    if rh.size:
        f = np.fft.rfft(rh, axis=0)
        fr = np.fft.rfftfreq(rh.shape[0], cfg.policy_dt)
        f[fr < 2.0] = 0
        hf = np.sqrt(np.mean(np.fft.irfft(f, rh.shape[0], axis=0) ** 2))
    else:
        hf = np.nan
    cnt = np.maximum(acc["cnt"], 1)
    up = acc["cnt"] > 0
    deg = 180 / np.pi
    return {
        "fell": float(fell.mean()),
        "speed": float(np.sqrt((acc["v2"][up] / cnt[up]).mean())) if up.any() else np.nan,
        "rate": float(np.sqrt((acc["r2"][up] / cnt[up]).mean()) * deg) if up.any() else np.nan,
        "hf": float(hf * deg),
        "dcmd": float(np.sqrt((acc["du2"][up] / cnt[up]).mean())) if up.any() else np.nan,
        "sat": float((acc["sat"][up] / cnt[up]).mean()) if up.any() else np.nan,
        "creep": float(np.abs(acc["v1"][up] / cnt[up]).mean()) if up.any() else np.nan,
    }


ROWS = {
    "nominal": (Randomization(), 0.0),
    "25% ranges": (Randomization(), 0.25),
    "50% ranges": (Randomization(), 0.5),
    "full ranges": (Randomization(), 1.0),
    "full, IMU lat 80-120ms": (replace(Randomization(), imu_latency=__import__("bizbot_mj").U(0.08, 0.12, 0.1)), 1.0),
    "nominal, IMU lat 60ms": (fixed(imu_latency=0.06), 0.0),
    "nominal, flex 4Hz": (fixed(flex=1, flex_hz=4.0, flex_zeta=0.03), 0.0),
    "nominal, mu 0.15": (fixed(mu=0.15), 0.0),
}


def fmt(r):
    return (f"{r['fell'] * 100:5.1f}% {r['speed']:6.3f} {r['rate']:6.2f} {r['hf']:6.2f} "
            f"{r['dcmd']:6.3f} {r['sat'] * 100:5.1f}%")


HEADER = "  fell  speed   rate     hf   dcmd    sat"
