"""Watch one robot in the MuJoCo viewer, driven by a trained policy or an LQR.

    .venv/bin/mjpython model/mj/watch.py [--policy model/mj/runs/<run>/final.pt]
    .venv/bin/mjpython model/mj/watch.py --controller "tuned LQR" --scale 0.5

Real time. Each robot is drawn from the ranges at --scale (0 nominal, 1 full);
its key values print when it appears. Push it yourself: double-click the body
to select it, then Ctrl + right-drag. Enter: new random robot. Random pushes
are off unless --pushes, random slope/payload/floor changes unless --changes.
A robot that passes 35 degrees is replaced.

Type in the terminal while it runs, to change this robot live:
    com <cm>     centre of mass forward (+) or back (-) of the body axis
    lat <ms>     IMU latency (up to 250)
    tilt <deg>   IMU mounting tilt
    show         print the current values;  new  new random robot

mjpython is needed on macOS for the viewer window.
"""

import argparse
import queue
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import mujoco
import mujoco.viewer
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
KEY_ENTER = 257
MAX_LATENCY_S = 0.25


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, default=None)
    ap.add_argument("--controller", choices=["untuned LQR", "tuned LQR", "firmware PID"], default=None)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pushes", action="store_true")
    ap.add_argument("--changes", action="store_true")
    args = ap.parse_args()

    rand = Randomization()
    if not args.pushes:
        rand = replace(rand, push_rate=U(0.0, 0.0, 0.0))
    if not args.changes:
        rand = replace(rand, change_rate=U(0.0, 0.0, 0.0))
    if args.controller:
        import metrics
        env = BalanceEnv(EnvConfig(n_envs=1, policy_dt=0.005, action_rate_limit=2.0, episode_s=1e9), rand,
                         seed=args.seed, scale=args.scale)
        ctrl = metrics.classical(args.controller, 1, env)
        name = args.controller
        act = lambda obs: ctrl(env.plant.measurements(), obs)
        remake = lambda: metrics.classical(args.controller, 1, env)
    else:
        from ppo import load_policy
        path = args.policy or DEFAULT
        policy, ck = load_policy(path)
        e = {k: v for k, v in ck["env"].items() if k not in ("n_envs", "episode_s")}
        env = BalanceEnv(EnvConfig(n_envs=1, episode_s=1e9, **e), rand, seed=args.seed, scale=args.scale)
        name = f"policy {path.parent.name}/{path.name}"
        act = policy
        remake = None
    pl = env.plant
    m = pl.m
    drive = None if args.controller else DriveController(env.cfg.policy_dt, env.cfg.history,
                                                         env.cfg.integral_leak_s, env.cfg.integral_limit)
    # Draw the skin instead of the plain physics shapes (which still collide and count).
    m.geom_rgba[pl.g_shell, 3] = m.geom_rgba[pl.g_wl, 3] = m.geom_rgba[pl.g_wr, 3] = 0.0
    up_vis = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"vis_{k}") for k, _ in UPPER])
    up_pos = m.geom_pos[up_vis].copy()

    def place_skin():
        """Keep the skin's height above the axle whatever this robot's flex hinge height is."""
        m.geom_pos[up_vis] = up_pos - np.array([0.0, 0.0, m.body_pos[pl.b_up[0], 2]])

    place_skin()

    # Room in the delay lines for latencies beyond the training range.
    H = int(np.ceil(MAX_LATENCY_S / pl.dt)) + 2
    for name in ("hist_cmd", "hist_pitch", "hist_accel", "hist_rpm"):
        old = getattr(pl, name)
        ordered = np.concatenate([old[pl.ptr:], old[:pl.ptr]])      # oldest first
        new = np.repeat(ordered[:1], H, axis=0)
        new[H - len(ordered):] = ordered
        setattr(pl, name, new)
    pl.H, pl.ptr = H, 0
    turn = None
    if drive is not None:
        pl = TurningPlant.from_plant(pl, env.cfg.max_cmd)
        env.plant = pl
        turn = TurnController(env.cfg.policy_dt)

    commands = queue.Queue()

    def read_stdin():
        for line in sys.stdin:
            commands.put(line.strip())

    threading.Thread(target=read_stdin, daemon=True).start()

    def apply(cmd):
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
                force, duration = float(parts[1]), float(parts[2])
                if not np.isfinite(force) or not -150 <= force <= 150 or not .05 <= duration <= 2:
                    raise ValueError("Invalid push")
                pl.push_F[0] = force
                pl.push_h[0] = .20
                pl.push_until[0] = pl.t[0] + duration
                return
            elif parts[0] == "new":
                want_new["flag"] = True
                return
            elif parts[0] != "show":
                print("  commands: com <cm>, lat <ms>, tilt <deg>, show, new", flush=True)
                return
        except (IndexError, ValueError):
            print("  usage: com <cm> | lat <ms> | tilt <deg>", flush=True)
            return
        print(f"  now: COM {p['com_offset'][0] * 100:+.1f} cm, IMU latency {p['imu_latency'][0] * 1000:.0f} ms, "
              f"IMU tilt {p['imu_mount_deg'][0]:+.1f} deg", flush=True)

    want_new = {"flag": False}

    def key(k):
        if k == KEY_ENTER:
            want_new["flag"] = True

    def describe():
        p = pl.p
        print(f"[{name}] robot: mass {p['body_mass'][0]:.0f} kg, COM {p['com_height'][0]:.2f} m up "
              f"{p['com_offset'][0] * 100:+.1f} cm fwd, IMU latency {p['imu_latency'][0] * 1000:.0f} ms, "
              f"IMU offset {p['imu_mount_deg'][0]:+.1f} deg, floor mu {p['mu'][0]:.2f}, slope {p['slope_deg'][0]:+.1f} deg, "
              f"flex {'%.0f Hz' % p['flex_hz'][0] if p['flex'][0] else 'none'}", flush=True)

    describe()
    obs = env._obs()
    panel = ControlPanel(commands)
    panel.publish(pl, drive, turn)
    with panel, mujoco.viewer.launch_passive(pl.m, pl.d, key_callback=key) as v:
        with v.lock():
            v.cam.distance, v.cam.elevation, v.cam.azimuth = 1.65, -8.0, 205.0
            follow_robot(m, pl.d, v.cam)
        t_next = time.perf_counter()
        next_publish = 0.0
        while v.is_running():
            while not commands.empty():
                with v.lock():
                    apply(commands.get())
            if want_new["flag"]:
                want_new["flag"] = False
                env._reset(np.arange(1))
                place_skin()
                obs = env._obs()
                if drive is not None:
                    drive.reset(require_release=True)
                    turn.reset()
                if remake:
                    ctrl = remake()
                    act = lambda o: ctrl(pl.measurements(), o)
                describe()
            with v.lock():
                pl.d.xfrc_applied[:, 1:] = 0.0
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
                obs, _, term, _, _ = env.step(np.asarray(act(policy_obs)).reshape(1))
                if drive is not None:
                    if term[0]:
                        drive.reset(require_release=True)
                        turn.reset()
                    else:
                        drive.record(obs, pl.p['wheel_radius'][0])
            if term[0]:
                print("  fell; new robot", flush=True)
                place_skin()
                if remake:
                    ctrl = remake()
                    act = lambda o: ctrl(pl.measurements(), o)
                describe()
            # Clear last frame's mouse push so it can't pile up; sync applies this frame's.
            pl.d.xfrc_applied[:] = 0.0
            with v.lock():
                follow_robot(m, pl.d, v.cam)
            v.sync()
            if time.perf_counter() >= next_publish:
                panel.publish(pl, drive, turn)
                next_publish = time.perf_counter() + .1
            t_next += env.cfg.policy_dt
            time.sleep(max(0.0, t_next - time.perf_counter()))


if __name__ == "__main__":
    main()
