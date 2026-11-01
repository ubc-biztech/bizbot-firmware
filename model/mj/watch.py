"""Watch one robot in MuJoCo, driven by a trained policy or an LQR, in one browser tab.

    .venv/bin/python model/mj/watch.py [--policy model/mj/runs/<run>/final.pt] [--robot <fitted robot>.json]
    .venv/bin/python model/mj/watch.py --mode PID --pid 0.062 0.2 0.0018 \
        --policy "RL old=model/mj/runs/ft_merged/ckpt_0040M.pt" --policy "RL best=model/mj/runs/fit_ft_I/ckpt_0080M.pt"
    .venv/bin/python model/mj/watch.py --controller "tuned LQR" --scale 0.5

Real time. The tab shows the 3D view (drag to orbit, scroll to zoom) next to the
controls: switch between the firmware PID (blue, gains editable live), Daniel's
PID from origin/daniel-updates with his tuner values (green, fixed), and each RL
policy given with --policy (red; orange when its label says "old"), drive with
W/A/S/D (RL only), push the torso with I/J/K/L
forward/left/back/right relative to where the robot faces, live settings, new
robot. Switching controller restarts the robot from a still start. Each robot is drawn from the ranges at --scale (0 nominal, 1 full); its key
values show under the view and print when it appears. Random pushes are off unless
--pushes, random slope/payload/floor changes unless --changes. A robot that passes
35 degrees is replaced. Closing the tab (or Ctrl-C) ends the program.

Type in the terminal while it runs, to change this robot live:
    com <cm>     centre of mass forward (+) or back (-) of the body axis
    lat <ms>     IMU latency (up to 250)
    tilt <deg>   IMU mounting tilt
    show         print the current values;  new  new random robot
    mode <tab>   switch controller (PID or a policy label);  pid <kp> <ki> <kd>  PID gains
"""

import argparse
import json
import queue
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "sim"))
sys.path.insert(0, str(HERE))
import bizbot_mj.plant as plant_mod  # noqa: E402
from bizbot_mj import Randomization, U  # noqa: E402
from bizbot_mj.env import BalanceEnv, EnvConfig  # noqa: E402
from controls import ControlPanel  # noqa: E402
from drive import DriveController  # noqa: E402
from turn import TurnController  # noqa: E402
from turning_plant import TurningPlant  # noqa: E402

DEFAULT = HERE / "runs" / "r4" / "td3-full-l1-h40-2upd-s2" / "final.pt"
MAX_LATENCY_S = 0.25
VIEW_W, VIEW_H, VIEW_FPS = 960, 600, 60


def rounded_box(name, hx, hy, hz, e=0.25, nu=16, nv=24):
    """Mesh asset: a superellipsoid with half sizes hx, hy, hz; small e is boxier."""
    pts = []
    sg = lambda x: np.sign(x) * np.abs(x) ** e
    for i in range(nu + 1):
        u = -np.pi / 2 + np.pi * i / nu
        for j in range(nv):
            v = -np.pi + 2 * np.pi * j / nv
            pts.append((hx * sg(np.cos(u)) * sg(np.cos(v)), hy * sg(np.cos(u)) * sg(np.sin(v)), hz * sg(np.sin(u))))
    vert = " ".join(f"{x:.4f} {y:.4f} {z:.4f}" for x, y, z in pts)
    return f'<mesh name="{name}" vertex="{vert}"/>'


# Visual-only parts (no mass, no contacts), in metres from the axle; +x is forward.
# The bodies keep their explicit inertials, so none of this changes the physics.
SKIN = dict(body=".55 .56 .65 1", white=".86 .87 .89 1", visor=".012 .018 .025 1", eye=".48 .85 1 1",
            tire=".025 .027 .03 1", hub=".32 .48 .54 1")
UPPER = [  # (name, geom attributes) on the upper body; z is shifted by the flex hinge height at each reset
    ("torso", 'type="mesh" mesh="m_torso" pos="0 0 0.20" material="skin_body"'),
    ("head", 'type="mesh" mesh="m_head" pos="0 0 0.445" material="skin_body"'),
    ("visor", 'type="mesh" mesh="m_visor" pos="0.094 0 0.435" material="skin_visor"'),
    ("eye_l", 'type="ellipsoid" size="0.004 0.014 0.016" pos="0.113 0.047 0.449" material="skin_eye"'),
    ("eye_r", 'type="ellipsoid" size="0.004 0.014 0.016" pos="0.113 -0.047 0.449" material="skin_eye"'),
    ("ear_l", 'type="mesh" mesh="m_ear" pos="0 0.153 0.443" material="skin_body"'),
    ("ear_r", 'type="mesh" mesh="m_ear" pos="0 -0.153 0.443" material="skin_body"'),
    ("ant_l", 'type="capsule" size="0.008" fromto="0 0.153 0.49 0 0.153 0.57" material="skin_body"'),
    ("ant_r", 'type="capsule" size="0.008" fromto="0 -0.153 0.49 0 -0.153 0.57" material="skin_body"'),
    ("shoulder_l", 'type="capsule" size="0.015" fromto="0 0.14 0.29 0 0.185 0.275" material="skin_body"'),
    ("shoulder_r", 'type="capsule" size="0.015" fromto="0 -0.14 0.29 0 -0.185 0.275" material="skin_body"'),
    ("arm_l", 'type="mesh" mesh="m_arm" pos="0 0.192 0.255" euler="15 0 0" material="skin_body"'),
    ("arm_r", 'type="mesh" mesh="m_arm" pos="0 -0.192 0.255" euler="-15 0 0" material="skin_white"'),
] + [
    (f"smile{k}", f'type="capsule" size="0.003" fromto="0.114 {0.022 * np.sin(a):.5f} {0.410 + 0.013 * (1 - np.cos(a)):.5f} '
                  f'0.114 {0.022 * np.sin(b):.5f} {0.410 + 0.013 * (1 - np.cos(b)):.5f}" '
                  f'material="skin_eye"') for k, (a, b) in enumerate(zip(np.linspace(-1.1, 1.1, 13)[:-1], np.linspace(-1.1, 1.1, 13)[1:]))]


def visual_xml(n, dt):
    """The plant's model with a checkered floor, a light and the robot's skin; physics unchanged."""
    xml = _xml_orig(n, dt)
    mats = "".join(f'<material name="skin_{k}" rgba="{v}" specular=".3" shininess=".4"'
                   + (' emission="1"' if k == "eye" else "") + "/>" for k, v in SKIN.items())
    xml = xml.replace('<mujoco model="bizbot">', '<mujoco model="bizbot"><asset>'
                      '<texture name="grid" type="2d" builtin="checker" rgb1=".32 .36 .42" rgb2=".22 .25 .3" '
                      'width="512" height="512"/><material name="grid" texture="grid" texrepeat="4 4" '
                      'texuniform="true" reflectance=".05"/><texture type="skybox" builtin="gradient" '
                      'rgb1=".55 .68 .82" rgb2=".1 .12 .16" width="256" height="256"/>' + mats
                      + rounded_box("m_torso", 0.085, 0.145, 0.155, e=0.28, nu=32, nv=48)
                      + rounded_box("m_head", 0.10, 0.135, 0.115, e=0.8, nu=32, nv=48)
                      + rounded_box("m_visor", 0.018, 0.100, 0.077, e=0.65, nu=32, nv=48)
                      + rounded_box("m_ear", 0.043, 0.023, 0.049, e=0.25)
                      + rounded_box("m_arm", 0.037, 0.031, 0.054, e=0.35)
                      + rounded_box("m_skirt", 0.085, 0.15, 0.03, e=0.4) + '</asset>', 1)
    xml = xml.replace('type="plane" size="0 0 1"/>', 'type="plane" size="0 0 1" material="grid"/>')
    xml = xml.replace("<worldbody>", '<worldbody><light pos="0 -1 3" dir="0 .3 -1" directional="true"/>', 1)
    up = "".join(f'<geom name="vis_{k}" {a} mass="0"/>' for k, a in UPPER)
    xml = xml.replace('<site name="imu_u0"', up + '<site name="imu_u0"', 1)
    xml = xml.replace('<site name="imu_b0"', '<geom name="vis_skirt" type="mesh" mesh="m_skirt" pos="0 0 0.0" '
                      'material="skin_white" mass="0"/><site name="imu_b0"', 1)
    for side, sgn in (("l", 1), ("r", -1)):
        xml = xml.replace(f'<geom name="g{side}0"', f'<geom name="vis_tire_{side}" type="cylinder" size="0.0825 0.045" '
                          f'pos="0 {-sgn * 0.04} 0" zaxis="0 1 0" material="skin_tire" mass="0"/><geom name="vis_hub_{side}" type="cylinder" '
                          f'size="0.04 0.006" pos="0 {sgn * 0.006} 0" zaxis="0 1 0" material="skin_hub" mass="0"/>'
                          f'<geom name="g{side}0"', 1)
    return xml


_xml_orig = plant_mod._xml
plant_mod._xml = visual_xml


def follow_robot(model, data, camera):
    mujoco.mj_kinematics(model, data)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = data.geom("vis_torso").xpos


BODY_COLOR = {"PID": ".18 .45 .85 1", "PID Daniel": ".15 .62 .4 1"}   # policies: red, or orange when the label says "old"; LQRs keep SKIN["body"]
RED, ORANGE = ".82 .2 .2 1", ".93 .5 .12 1"


def body_color(mode, policies):
    if mode in policies:
        return ORANGE if "old" in mode.lower() else RED
    return BODY_COLOR.get(mode)


def css(rgba):
    return "#" + "".join(f"{int(round(float(x) * 255)):02x}" for x in rgba.split()[:3]) if rgba else None
PID_ROBOT = (0.062, 0.2, 0.0018)   # the gains the robot ran (SET_PID 0.062 0.2 0.0018)


def build(mode, args, rand, gains, policies):
    """One controller driving one robot: its env, plant, step function, and (RL only) drive and turn helpers.

    PID runs like the firmware: 5 ms ticks on a 1 ms physics step, output cap MAX_BALANCE_OUTPUT 1.0 and
    rate filter IMU_RATE_LPF_ALPHA 0.15. RL runs with its checkpoint's own settings (including DZ_COMP)."""
    from bizbot_sim.controllers import daniel_pid, firmware_pid
    s = SimpleNamespace(mode=mode, drive=None, turn=None, remake=None, ctrl=None)
    still = dict(init_pitch_deg=1.0, init_rate_dps=0.0, init_speed=0.0) if args.still_start else {}
    if mode in policies:
        from ppo import load_policy
        path = policies[mode]
        policy, ck = load_policy(path)
        e = {k: v for k, v in ck["env"].items() if k not in ("n_envs", "episode_s")}
        s.env = BalanceEnv(EnvConfig(n_envs=1, episode_s=1e9, **{**e, **still}), rand, seed=args.seed, scale=args.scale)
        s.name = f"{mode} · {path.parent.name}/{path.name}"
        s.act = policy
    else:
        if mode in ("PID", "PID Daniel"):
            s.env = BalanceEnv(EnvConfig(n_envs=1, policy_dt=0.005, physics_dt=0.001, action_rate_limit=2.0, episode_s=1e9, **still),
                               rand, seed=args.seed, scale=args.scale)
            if mode == "PID":
                pid = lambda: firmware_pid(1, kp=gains[0], ki=gains[1], kd=gains[2], cap=1.0, alpha=0.15)
                s.name = f"PID {gains[0]:g} / {gains[1]:g} / {gains[2]:g}"
            else:   # origin/daniel-updates with Daniel's tuner values (fixed)
                pid = lambda: daniel_pid(1)
                s.name = "PID Daniel 0.06/0.064 · 0 · 0.0015 · boost 1/0.8 · vel 0.09"
            s.remake = lambda: (lambda c: lambda m_, obs: c(m_))(pid())
        else:
            import metrics
            s.env = BalanceEnv(EnvConfig(n_envs=1, policy_dt=0.005, action_rate_limit=2.0, episode_s=1e9, **still), rand,
                               seed=args.seed, scale=args.scale)
            s.remake = lambda: metrics.classical(mode, 1, s.env)
            s.name = mode
        s.ctrl = s.remake()
        s.act = lambda obs: s.ctrl(s.env.plant.measurements(), obs)
    pl = s.env.plant
    m = pl.m
    if mode in policies:
        s.drive = DriveController(s.env.cfg.policy_dt, s.env.cfg.history, s.env.cfg.integral_leak_s, s.env.cfg.integral_limit)
    # Draw the skin instead of the plain physics shapes (which still collide and count).
    m.geom_rgba[pl.g_shell, 3] = m.geom_rgba[pl.g_wl, 3] = m.geom_rgba[pl.g_wr, 3] = 0.0
    color = body_color(mode, policies)
    if color:
        m.mat_rgba[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "skin_body")] = [float(x) for x in color.split()]
    s.up_vis = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"vis_{k}") for k, _ in UPPER])
    s.up_pos = m.geom_pos[s.up_vis].copy()
    # Room in the delay lines for latencies beyond the training range.
    H = int(np.ceil(MAX_LATENCY_S / pl.dt)) + 2
    for buf in ("hist_cmd", "hist_pitch", "hist_accel", "hist_rpm", "hist_wheel_rpm"):
        old = getattr(pl, buf)
        ordered = np.concatenate([old[pl.ptr:], old[:pl.ptr]])      # oldest first
        new = np.repeat(ordered[:1], H, axis=0)
        new[H - len(ordered):] = ordered
        setattr(pl, buf, new)
    pl.H, pl.ptr = H, 0
    if s.drive is not None:
        pl = TurningPlant.from_plant(pl, s.env.cfg.max_cmd)
        s.env.plant = pl
        s.turn = TurnController(s.env.cfg.policy_dt)
    s.pl, s.m = pl, m
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", action="append", default=None, metavar="[LABEL=]PATH",
                    help="an RL policy tab; repeat for several, e.g. --policy 'RL old=...pt' --policy 'RL best=...pt' "
                         "(a bare path is labelled RL)")
    ap.add_argument("--controller", choices=["untuned LQR", "tuned LQR", "firmware PID"], default=None,
                    help="start with this controller ('firmware PID' = the PID mode with --pid gains)")
    ap.add_argument("--mode", default=None, help="tab to start on: PID or a policy label (default: the first policy)")
    ap.add_argument("--pid", type=float, nargs=3, default=list(PID_ROBOT), metavar=("KP", "KI", "KD"))
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pushes", action="store_true")
    ap.add_argument("--changes", action="store_true")
    ap.add_argument("--robot", type=Path, default=None,
                    help="JSON of parameter values (a fitted robot): every robot drawn is that one, at scale 0")
    ap.add_argument("--still-start", action="store_true",
                    help="each robot starts within +-1 deg and not moving, like letting go of the real robot "
                         "(default: up to 5 deg, 20 deg/s and 0.3 m/s)")
    args = ap.parse_args()

    rand = Randomization()
    if args.robot:
        from bizbot_mj import fixed
        rand = fixed(**json.load(open(args.robot)))
        args.scale = 0.0
    if not args.pushes:
        rand = replace(rand, push_rate=U(0.0, 0.0, 0.0))
    if not args.changes:
        rand = replace(rand, change_rate=U(0.0, 0.0, 0.0))
    policies = {}
    for item in args.policy or [str(DEFAULT)]:
        label, _, path = item.rpartition("=")
        policies[label.strip() or "RL"] = Path(path)
    modes = ["PID", "PID Daniel"] + list(policies)
    first = args.mode or next(iter(policies))
    if args.controller == "firmware PID":
        first = "PID"
    elif args.controller:
        modes.append(args.controller)
        first = args.controller
    if first not in modes:
        ap.error(f"--mode must be one of {modes}")
    gains = list(args.pid)
    s = build(first, args, rand, gains, policies)

    def place_skin():
        """Keep the skin's height above the axle whatever this robot's flex hinge height is."""
        s.m.geom_pos[s.up_vis] = s.up_pos - np.array([0.0, 0.0, s.m.body_pos[s.pl.b_up[0], 2]])

    place_skin()

    commands = queue.Queue()
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 1.65, -8.0, 205.0

    def read_stdin():
        for line in sys.stdin:
            commands.put(line.strip())

    threading.Thread(target=read_stdin, daemon=True).start()
    want = {"new": False, "mode": None}

    def apply(cmd):
        pl, drive, turn = s.pl, s.drive, s.turn
        p = pl.p
        parts = cmd.split()
        if not parts:
            return
        try:
            if parts[0] == "speed_limit":
                if drive is not None:
                    drive.set_speed_limit(float(parts[1]))
                return
            elif parts[0] == "acceleration_limit":
                if drive is not None:
                    drive.set_acceleration_limit(float(parts[1]))
                return
            elif parts[0] == "turn_limit":
                if turn is not None:
                    turn.set_rate_limit(float(parts[1]))
                return
            elif parts[0] == "turn_acceleration":
                if turn is not None:
                    turn.set_acceleration_limit(float(parts[1]))
                return
            elif parts[0] == "turn_command_limit":
                if turn is not None:
                    turn.set_command_limit(float(parts[1]))
                return
            elif parts[0] == "com":
                p["com_offset"][0] = float(parts[1]) / 100.0
                pl._write_body(np.arange(1))
            elif parts[0] == "lat":
                p["imu_latency"][0] = np.clip(float(parts[1]), 0, MAX_LATENCY_S * 1000) / 1000.0
                p["imu_delay"][0] = int(np.rint(p["imu_latency"][0] / pl.dt))
            elif parts[0] == "tilt":
                p["imu_mount_deg"][0] = float(parts[1])
            elif parts[0] == "push":
                force, duration, direction = float(parts[1]), float(parts[2]), parts[3]
                if not np.isfinite(force) or not 1 <= force <= 150 or not .05 <= duration <= 2 \
                        or direction not in ("forward", "back", "left", "right"):
                    raise ValueError("Invalid push")
                key_push.update(force=force, direction=direction, until=pl.t[0] + duration)
                return
            elif parts[0] == "new":
                want["new"] = True
                return
            elif parts[0] == "mode":
                mode = " ".join(parts[1:])
                if mode not in modes:
                    raise ValueError("Invalid mode")
                want["mode"] = mode
                return
            elif parts[0] == "pid":
                kp, ki, kd = (float(x) for x in parts[1:4])
                if not all(np.isfinite([kp, ki, kd])) or not (0 <= kp <= 1 and 0 <= ki <= 10 and 0 <= kd <= 0.1):
                    raise ValueError("Invalid gains")
                gains[:] = [kp, ki, kd]
                if s.mode == "PID":
                    s.ctrl = s.remake()
                    s.name = f"PID {kp:g} / {ki:g} / {kd:g}"
                    describe()
                return
            elif parts[0] == "cam":
                cam.azimuth = (cam.azimuth + float(parts[1])) % 360.0
                cam.elevation = float(np.clip(cam.elevation + float(parts[2]), -89.0, 89.0))
                cam.distance = float(np.clip(cam.distance * np.exp(float(parts[3])), 0.3, 10.0))
                return
            elif parts[0] != "show":
                print("  commands: com <cm>, lat <ms>, tilt <deg>, show, new, mode <tab>, pid <kp> <ki> <kd>", flush=True)
                return
        except (IndexError, ValueError):
            print("  usage: com <cm> | lat <ms> | tilt <deg> | mode <tab> | pid <kp> <ki> <kd>", flush=True)
            return
        print(f"  now: COM {p['com_offset'][0] * 100:+.1f} cm, IMU latency {p['imu_latency'][0] * 1000:.0f} ms, "
              f"IMU tilt {p['imu_mount_deg'][0]:+.1f} deg", flush=True)

    key_push = {"force": 0.0, "direction": "", "until": 0.0}   # I/J/K/L push, in sim time

    def apply_key_push():
        """Hold this step's I/J/K/L push on the torso, along the robot's own forward or left axis."""
        pl = s.pl
        pl.d.qfrc_applied[:] = 0.0
        if pl.t[0] >= key_push["until"]:
            return
        hx, hy = pl.heading()[0]
        fwd, left = np.array([hx, hy, 0.0]), np.array([-hy, hx, 0.0])
        axis = {"forward": fwd, "back": -fwd, "left": left, "right": -left}[key_push["direction"]]
        mujoco.mj_applyFT(s.m, pl.d, key_push["force"] * axis, np.zeros(3), pl.d.geom("vis_torso").xpos.copy(),
                          int(pl.b_up[0]), pl.d.qfrc_applied)

    def key_push_state():
        return ({"force": key_push["force"], "direction": key_push["direction"]}
                if s.pl.t[0] < key_push["until"] else None)

    def robot_info():
        p = s.pl.p
        return {"mass": float(p["body_mass"][0]), "com_height": float(p["com_height"][0]),
                "com_offset": float(p["com_offset"][0] * 100), "imu_latency": float(p["imu_latency"][0] * 1000),
                "imu_offset": float(p["imu_mount_deg"][0]), "mu": float(p["mu"][0]), "slope": float(p["slope_deg"][0]),
                "flex": float(p["flex_hz"][0]) if p["flex"][0] else None,
                "carpet": float(p["floor_spring"][0]) if p["floor_spring_on"][0] else None}

    def describe():
        p = s.pl.p
        text["robot"] = (f"[{s.name}] robot: mass {p['body_mass'][0]:.0f} kg, COM {p['com_height'][0]:.2f} m up "
                         f"{p['com_offset'][0] * 100:+.1f} cm fwd, IMU latency {p['imu_latency'][0] * 1000:.0f} ms, "
                         f"IMU offset {p['imu_mount_deg'][0]:+.1f} deg, floor mu {p['mu'][0]:.2f}, slope {p['slope_deg'][0]:+.1f} deg, "
                         f"flex {'%.0f Hz' % p['flex_hz'][0] if p['flex'][0] else 'none'}")
        print(text["robot"], flush=True)

    def extra():
        return {"mode": s.mode, "modes": modes, "colors": {k: css(body_color(k, policies)) for k in modes},
                "pid": list(gains), "name": s.name, "robot_info": robot_info()}

    text = {"robot": ""}
    describe()
    obs = s.env._obs()
    panel = ControlPanel(commands)
    panel.publish(s.pl, s.drive, s.turn, text["robot"], key_push_state(), extra())

    def make_renderer():
        s.m.vis.global_.offwidth, s.m.vis.global_.offheight = VIEW_W, VIEW_H
        return mujoco.Renderer(s.m, VIEW_H, VIEW_W)

    renderer = make_renderer()
    with panel:
        t_next = time.perf_counter()
        next_publish = next_frame = 0.0
        while not panel.page_closed():
            while not commands.empty():
                apply(commands.get())
            if want["mode"] is not None:
                # New controller: rebuild the robot for its tick rate (fresh still start), keep the live settings.
                mode, want["mode"] = want["mode"], None
                keep = {k: float(s.pl.p[k][0]) for k in ("com_offset", "imu_mount_deg", "imu_latency")}
                renderer.close()
                s = build(mode, args, rand, gains, policies)
                for k, v in keep.items():
                    s.pl.p[k][0] = v
                s.pl.p["imu_delay"][0] = int(np.rint(keep["imu_latency"] / s.pl.dt))
                s.pl._write_body(np.arange(1))
                place_skin()
                renderer = make_renderer()
                key_push["until"] = 0.0
                obs = s.env._obs()
                t_next = time.perf_counter()
                describe()
            if want["new"]:
                want["new"] = False
                key_push["until"] = 0.0
                s.env._reset(np.arange(1))
                place_skin()
                obs = s.env._obs()
                if s.drive is not None:
                    s.drive.reset(require_release=True)
                    s.turn.reset()
                if s.remake:
                    s.ctrl = s.remake()
                describe()
            pl, drive, turn, env = s.pl, s.drive, s.turn, s.env
            pl.d.xfrc_applied[:, 1:] = 0.0
            apply_key_push()
            if drive is not None:
                direction, turn_request = panel.inputs()
                turning = bool(turn_request or turn.engaged)
                policy_obs = drive.adapt(obs, pl.p['wheel_radius'][0], direction,
                                         turning=turning)
                pl.turn_command[0] = turn.update(turn_request, drive, pl.fb_rpm[0, 0], pl.fb_rpm[1, 0],
                                                 pl.p['wheel_radius'][0], pl.p['track'][0],
                                                 pl.p['speed_full_scale_rpm'][0], env.cfg.max_cmd)
            else:
                policy_obs = obs
            obs, _, term, _, _ = env.step(np.asarray(s.act(policy_obs)).reshape(1))
            if drive is not None:
                if term[0]:
                    drive.reset(require_release=True)
                    turn.reset()
                else:
                    drive.record(obs, pl.p['wheel_radius'][0])
            if term[0]:
                key_push["until"] = 0.0
                pl.d.qfrc_applied[:] = 0.0
                print("  fell; new robot", flush=True)
                place_skin()
                if s.remake:
                    s.ctrl = s.remake()
                describe()
            pl.d.xfrc_applied[:] = 0.0
            if time.perf_counter() >= next_frame:
                follow_robot(s.m, pl.d, cam)
                renderer.update_scene(pl.d, camera=cam)
                panel.show(renderer.render())
                next_frame = max(next_frame + 1.0 / VIEW_FPS, time.perf_counter() - 1.0 / VIEW_FPS)
            if time.perf_counter() >= next_publish:
                panel.publish(pl, drive, turn, text["robot"], key_push_state(), extra())
                next_publish = time.perf_counter() + .1
            t_next += env.cfg.policy_dt
            time.sleep(max(0.0, t_next - time.perf_counter()))
    renderer.close()


if __name__ == "__main__":
    main()
