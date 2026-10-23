"""Live sim for sim/viewer.html.

    .venv/bin/python sim/live.py        # then open http://localhost:8765
    .venv/bin/python sim/live.py --policy model/runs/<run>/final.zip   # adds "RL policy"

One robot, stepped on request from the page. The page picks the controller and
the robot's conditions, and can push the robot at any time. The controller sees
only the ESP32's measurements.
"""

import argparse
import json
import sys
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bizbot_sim import BizBotEnv, EnvConfig, Plant, Randomization, U  # noqa: E402
from bizbot_sim.controllers import CONTROL_DT, CONTROLLERS, MAX_CMD  # noqa: E402
from bizbot_sim.plant import DEG, RPM_PER_RAD_S  # noqa: E402

PORT = 8765
PHYSICS_DT = 0.00025
SUBSTEPS = int(round(CONTROL_DT / PHYSICS_DT))
RECORD_EVERY = 2         # frame every other control tick: 100 Hz
MAX_STEP_S = 0.5         # longest stretch one request may ask for
MAX_TILT_DEG = 35.0      # firmware MAX_TILT_DEG
MODES = ("voltage", "speed", "torque")
HTML = Path(__file__).parent / "viewer.html"

NOTES = {
    "none": "Command 0 all the time. In speed mode the hoverboard then holds the wheels still, "
            "so the robot should fall in well under a second.",
    "pid": "The HTML sim's cascaded PID with its default gains (kp 0.12, ki 0.664, kd 0.015, "
           "vkp 0.025, vki 0.075, output cap 0.45): a wheel-rpm PI sets the target pitch and a "
           "pitch PID sets the command. Pitch rate is the differenced IMU pitch, low-passed "
           "with alpha 0.15.",
    "lqr": "Fixed-gain LQR designed for the nominal robot (20 kg, COM 0.45 m, speed mode, "
           "stock hoverboard filter). A Kalman filter estimates pitch, pitch rate, speed and the "
           "hoverboard's speed-loop integral from the delayed IMU pitch and wheel rpm. Position is "
           "integrated from speed, so it drives back to where it started. It does not know which "
           "robot it is on.",
}

# Set by --policy: {"model": SB3 PPO, "env": its run's EnvConfig, "label": menu name}.
POLICY = None

# Nominal robot with no random pushes or random condition changes.
QUIET = replace(Randomization(), push_rate=U(0.0, 0.5, 0.0), change_rate=U(0.0, 0.2, 0.0))

_nom = lambda k: getattr(Randomization(), k).nominal
DEFAULT_CONDITIONS = {
    "random": 0.0, "mu": _nom("mu_kinetic"), "slope_deg": _nom("slope_deg"),
    "mass": _nom("body_mass"), "com_height": _nom("com_height"),
    "com_forward_cm": _nom("com_offset") * 100, "imu_latency_ms": _nom("imu_latency") * 1000,
    "imu_noise_deg": _nom("imu_noise_deg"), "mode": "speed", "filter": "stock",
    "battery_v": _nom("battery_ocv"), "start_tilt_deg": 3.0,
}

# Recorded channels and their rounding (decimals).
CHANNELS = {
    "t": 3, "x": 4, "pitch": 3, "pitch_meas": 2, "wheel_angle": 3, "slip": 4, "speed": 3,
    "cmd_sent": 0, "cmd_motor": 1, "rpm_true": 1, "rpm_left": 0, "rpm_right": 0,
    "push": 1, "push_h": 3, "battery_v": 2, "battery_a": 1,
    "slope": 2, "com_height": 3, "com_offset": 4, "body_mass": 1,
}


def robot_ranges(c):
    """Conditions from the page -> (Randomization, scale)."""
    if float(c["random"]) > 0:
        return Randomization(), float(c["random"])
    probs = [0.0, 0.0, 0.0]
    probs[MODES.index(c["mode"])] = 1.0
    pinned = {
        "mu_kinetic": c["mu"], "slope_deg": c["slope_deg"], "body_mass": c["mass"],
        "com_height": c["com_height"], "com_offset": c["com_forward_cm"] / 100.0,
        "imu_latency": c["imu_latency_ms"] / 1000.0, "imu_noise_deg": c["imu_noise_deg"],
        "battery_ocv": c["battery_v"],
    }
    if c["filter"] == "off":
        pinned.update(cmd_rate_limit=1e6, cmd_lowpass=1.0)
    return replace(QUIET, mode_probs=tuple(probs),
                   **{k: U(float(v), float(v), float(v)) for k, v in pinned.items()}), 0.0


def params_table(pl, i):
    p = pl.p
    f = lambda k: float(p[k][i])
    lr = lambda k, fmt: f"{fmt.format(p[k][0, i])} / {fmt.format(p[k][1, i])}"
    rate = p["rate"][i]
    rows = [
        ["Body", "body mass", f"{f('body_mass'):.1f} kg"],
        ["Body", "COM height / offset", f"{f('com_height'):.2f} m / {f('com_offset') * 100:+.1f} cm"],
        ["Body", "radius of gyration", f"{f('body_gyration'):.2f} m"],
        ["Body", "wheel mass / inertia / radius",
         f"{f('wheel_mass'):.1f} kg / {f('wheel_inertia'):.3f} / {f('wheel_radius') * 100:.2f} cm"],
        ["Floor", "mu kinetic / static", f"{f('mu_kinetic'):.2f} / {f('mu_s'):.2f}"],
        ["Floor", "rolling resistance", f"{f('rolling_resistance'):.3f}"],
        ["Floor", "slope", f"{f('slope_deg'):+.1f} deg"],
        ["Drive", "friction Coulomb / viscous",
         f"{f('coulomb_friction'):.2f} N m / {f('viscous_friction'):.3f}"],
        ["Motors", "back-EMF L / R", lr("k_m", "{:.2f}") + " V s/rad"],
        ["Motors", "phase resistance L / R", lr("r_m", "{:.2f}") + " ohm"],
        ["Motors", "current limit L / R", lr("imax_m", "{:.1f}") + " A"],
        ["Battery", "open-circuit / resistance",
         f"{f('battery_ocv'):.1f} V / {f('battery_resistance'):.2f} ohm"],
        ["Hoverboard", "mode", MODES[int(p["mode"][i])]],
        ["Hoverboard", "cmd rate limit / low-pass",
         ("off" if not np.isfinite(rate) or rate > 1e5 else f"{rate:.0f} per 5 ms")
         + f" / {f('cmd_lowpass'):.2f}"],
        ["Hoverboard", "command latency", f"{f('cmd_latency') * 1000:.0f} ms"],
        ["Feedback", "period / latency / filter",
         f"{f('fb_period') * 1000:.0f} / {f('fb_latency') * 1000:.0f} / {f('fb_lag') * 1000:.0f} ms"],
        ["Feedback", "drop / min rpm", f"{f('fb_drop_prob') * 100:.1f} % / {f('fb_min_rpm'):.1f}"],
        ["Feedback", "gain L / R", lr("fb_gain", "{:.3f}")],
        ["IMU", "period / latency",
         f"{f('imu_period') * 1000:.1f} / {f('imu_latency') * 1000:.0f} ms"],
        ["IMU", "pitch noise / mount error", f"{f('imu_noise_deg'):.3f} / {f('imu_mount_deg'):+.2f} deg"],
        ["IMU", "accel tilt / fusion time", f"{f('imu_accel_tilt'):+.2f} / {f('imu_fusion_s'):.1f} s"],
        ["IMU", "height / accel noise", f"{f('imu_height'):.2f} m / {f('accel_noise'):.2f} m/s2"],
        ["IMU", "packet drop", f"{f('imu_drop_prob') * 100:.1f} %"],
        ["Random", "pushes / max force", f"{f('push_rate'):.2f} per s / {f('push_force_max'):.0f} N"],
        ["Random", "sudden changes", f"{f('change_rate'):.2f} per s"],
    ]
    if p["mode"][i] == MODES.index("speed"):
        rows.insert(13, ["Hoverboard", "speed loop kp / ki / full scale",
                         f"{f('speed_kp'):.2f} / {f('spd_ki'):.1f} / {f('speed_full_scale_rpm'):.0f} rpm"])
    return rows


class Session:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset("pid", DEFAULT_CONDITIONS)

    def reset(self, algo, conditions):
        c = {**DEFAULT_CONDITIONS, **conditions}
        rand, scale = robot_ranges(c)
        pitch = np.radians([float(c["start_tilt_deg"])])
        if algo == "rl":
            # The training env itself, so the policy sees exactly its training
            # observations. Never resets on its own; CMD_VEL held at 0.
            cfg = EnvConfig(**{**POLICY["env"], "n_envs": 1, "physics_dt": PHYSICS_DT,
                               "max_tilt_deg": 90.0, "episode_s": 1e9})
            self.env = env = BizBotEnv(cfg, rand, scale=scale)
            self.plant = pl = env.plant
            pl.reset(np.arange(1), pitch=pitch, rate=np.zeros(1), speed=np.zeros(1))
            env.u[:] = env.integral[:] = env.linear[:] = 0.0
            env.vref_next[:] = np.inf
            env.hist[:] = env._frame(slice(None))[:, None, :]
            self.obs = env._obs()
            self.ctrl = None
        else:
            self.env = None
            self.plant = pl = Plant(1, rand, PHYSICS_DT, np.random.default_rng(), scale)
            pl.reset(np.arange(1), pitch=pitch, rate=np.zeros(1), speed=np.zeros(1))
            self.ctrl = CONTROLLERS[algo][1](1)
        self.tick = 0
        self.u = np.zeros(1)
        self.fell = False
        p = pl.p
        return {
            "algo": algo, "note": NOTES[algo], "conditions": c,
            "wheel_radius": float(p["wheel_radius"][0]), "imu_height": float(p["imu_height"][0]),
            "com_height": float(p["com_height"][0]), "params": params_table(pl, 0),
        }

    def _snapshot(self, frames):
        pl, tr = self.plant, self.plant.truth()
        row = {
            "t": self.tick * CONTROL_DT, "x": tr["position"], "pitch": tr["pitch"] * DEG,
            "pitch_meas": pl.imu_pitch, "wheel_angle": tr["wheel_angle"], "slip": tr["slip"],
            "speed": tr["speed"], "cmd_sent": np.trunc(self.u * MAX_CMD), "cmd_motor": pl.u_f,
            "rpm_true": (pl.phid - pl.thd) * RPM_PER_RAD_S,
            "rpm_left": pl.fb_rpm[0], "rpm_right": pl.fb_rpm[1],
            "push": tr["push"], "push_h": pl.push_h, "battery_v": tr["battery_v"],
            "battery_a": tr["battery_a"], "slope": tr["slope"] * DEG,
            "com_height": pl.p["com_height"], "com_offset": pl.p["com_offset"],
            "body_mass": pl.p["body_mass"],
        }
        for k, d in CHANNELS.items():
            frames[k].append(round(float(np.asarray(row[k]).reshape(-1)[0]), d))

    def step(self, seconds):
        if self.env is not None:
            return self._step_policy(seconds)
        pl = self.plant
        frames = {k: [] for k in CHANNELS}
        for _ in range(int(round(min(seconds, MAX_STEP_S) / CONTROL_DT))):
            if self.fell:
                break
            self.u = np.clip(self.ctrl(pl.measurements()), -1.0, 1.0)
            if self.tick % RECORD_EVERY == 0:
                self._snapshot(frames)
            pl.advance(np.trunc(self.u * MAX_CMD), SUBSTEPS)
            self.tick += 1
            if abs(pl.truth()["pitch"][0]) > np.radians(MAX_TILT_DEG):
                self.fell = True
                self._snapshot(frames)
        return {"frames": frames, "fell": self.fell}

    def _step_policy(self, seconds):
        env, pl = self.env, self.plant
        frames = {k: [] for k in CHANNELS}
        ticks = int(round(env.cfg.policy_dt / CONTROL_DT))
        for _ in range(int(round(min(seconds, MAX_STEP_S) / env.cfg.policy_dt))):
            if self.fell:
                break
            action, _ = POLICY["model"].predict(self.obs, deterministic=True)
            action = np.clip(np.asarray(action, dtype=float).reshape(-1), -1.0, 1.0)
            lim = env.cfg.action_rate_limit
            if env.base is None:
                self.u = env.u + np.clip(action - env.u, -lim, lim)   # what env.step will send
            else:
                self.u = env.sent.copy()                              # LQR plus correction, last sent
            self._snapshot(frames)
            self.obs = env.step(action)[0]
            self.tick += ticks
            if abs(pl.truth()["pitch"][0]) > np.radians(MAX_TILT_DEG):
                self.fell = True
                self._snapshot(frames)
        return {"frames": frames, "fell": self.fell}

    def push(self, force, duration, height):
        pl = self.plant
        pl.push_F[:] = force
        pl.push_until[:] = pl.t + duration
        pl.push_h[:] = height
        return {}


SESSION = Session()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, separators=(",", ":")).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/viewer.html"):
            self._send(200, HTML.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/info":
            algos = [{"key": k, "label": label, "note": NOTES[k]} for k, (label, _) in CONTROLLERS.items()]
            if POLICY is not None:
                algos.append({"key": "rl", "label": POLICY["label"], "note": NOTES["rl"]})
            self._send(200, {"algorithms": algos, "defaults": DEFAULT_CONDITIONS})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            with SESSION.lock:
                if self.path == "/api/reset":
                    out = SESSION.reset(body["algo"], body.get("conditions", {}))
                elif self.path == "/api/step":
                    out = SESSION.step(float(body["seconds"]))
                elif self.path == "/api/push":
                    out = SESSION.push(float(body["force"]), float(body["duration"]), float(body["height"]))
                else:
                    return self._send(404, {"error": "not found"})
        except (KeyError, ValueError, TypeError) as e:
            return self._send(400, {"error": repr(e)})
        self._send(200, out)

    def log_message(self, *args):
        pass


def load_policy(path):
    """SB3 PPO zip; its run's config.json sits next to it or one folder up (checkpoints)."""
    global POLICY
    from stable_baselines3 import PPO
    configs = [d / "config.json" for d in (path.parent, path.parent.parent) if (d / "config.json").exists()]
    if not configs:
        sys.exit(f"no config.json next to {path} or in its parent folder")
    env = json.loads(configs[0].read_text())["env"]
    history = env["history"]
    residual = env.get("lqr_base", False)
    POLICY = {"model": PPO.load(path, device="cpu"), "env": env,
              "label": "LQR + RL" if residual else "RL policy"}
    NOTES["rl"] = (f"PPO policy {path.name} from {path.parent}. It sees the same sensor history "
                   f"it was trained on ({history} steps at 100 Hz) and is told to stand still."
                   + (f" Its output, times {env['residual_scale']}, is added to the LQR's command."
                      if residual else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", type=int, default=PORT)
    ap.add_argument("--policy", type=Path, help="trained policy .zip to add as 'RL policy'")
    args = ap.parse_args()
    if args.policy:
        load_policy(args.policy)
    port = args.port
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"BizBot live sim: http://localhost:{port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
