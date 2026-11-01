"""Balance controllers for the sim.

Each one sees only what the ESP32 sees (Plant.measurements()), runs every
CONTROL_DT, and returns a normalised command in [-1, 1], positive = forward,
which the firmware sends to the hoverboard as command * MAX_HOVERBOARD_COMMAND.
"""

import numpy as np

from .params import Randomization
from .plant import DEG, EFERU_FULL_SCALE, G, RPM_PER_RAD_S

CONTROL_DT = 0.005   # firmware control period
MAX_CMD = 500        # firmware MAX_HOVERBOARD_COMMAND


def no_control(n):
    return lambda m: np.zeros(n)


def cascaded_pid(n, kp=0.12, ki=0.664, kd=0.015, vkp=0.025, vki=0.075, cap=0.45,
                 alpha=0.15, vel_cap=3.0):
    """The HTML sim's controller: wheel-rpm PI -> target pitch -> pitch PID."""
    s = {"integ": np.zeros(n), "vint": np.zeros(n), "rate": np.zeros(n),
         "prev": None, "count": None}

    def ctrl(m):
        if s["prev"] is None:
            s["prev"], s["count"] = m["pitch_deg"].copy(), m["imu_count"].copy()
        new = m["imu_count"] != s["count"]
        raw = (m["pitch_deg"] - s["prev"]) / 0.01
        s["rate"] = np.where(new, s["rate"] + alpha * (raw - s["rate"]), s["rate"])
        s["prev"] = np.where(new, m["pitch_deg"], s["prev"])
        s["count"] = m["imu_count"].copy()
        verr = -0.5 * (m["rpm_left"] + m["rpm_right"])
        s["vint"] = np.clip(s["vint"] + verr * CONTROL_DT, -vel_cap / vki, vel_cap / vki)
        target = np.clip(vkp * verr + vki * s["vint"], -vel_cap, vel_cap)
        err = target - m["pitch_deg"]
        s["integ"] = np.clip(s["integ"] + err * CONTROL_DT, -cap / ki, cap / ki)
        out = np.clip(kp * err + ki * s["integ"] - kd * s["rate"], -cap, cap)
        return -out  # BALANCE_MOTOR_SIGN

    return ctrl


def firmware_pid(n, kp=0.04, ki=0.0, kd=0.001, cap=0.20, alpha=0.3, trim_deg=0.0):
    """firmware/src/main.cpp with the velocity loop off: angle PID around a trim.

    Pitch rate is Imu.cpp's: the pitch step since the last packet divided by
    the time since that packet was read (read every control tick), low-passed
    with alpha. The PID is PID.cpp: integral clamped to cap / ki, derivative
    on measurement, output clamped to +-cap, then BALANCE_MOTOR_SIGN.
    """
    s = {"integ": np.zeros(n), "rate": np.zeros(n), "prev": None, "count": None,
         "since": np.zeros(n)}

    def ctrl(m):
        if s["prev"] is None:
            s["prev"], s["count"] = m["pitch_deg"].copy(), m["imu_count"].copy()
        s["since"] += CONTROL_DT
        new = m["imu_count"] != s["count"]
        raw = (m["pitch_deg"] - s["prev"]) / np.maximum(s["since"], CONTROL_DT)
        s["rate"] = np.where(new, s["rate"] + alpha * (raw - s["rate"]), s["rate"])
        s["prev"] = np.where(new, m["pitch_deg"], s["prev"])
        s["since"] = np.where(new, 0.0, s["since"])
        s["count"] = m["imu_count"].copy()
        err = 0.0 - (m["pitch_deg"] - trim_deg)
        s["integ"] = s["integ"] + err * CONTROL_DT
        if ki != 0.0:
            s["integ"] = np.clip(s["integ"], -cap / abs(ki), cap / abs(ki))
        out = np.clip(kp * err + ki * s["integ"] - kd * s["rate"], -cap, cap)
        return -out  # BALANCE_MOTOR_SIGN

    return ctrl


def daniel_pid(n, kp_front=0.06, kp_back=0.064, ki=0.0, kd=0.0015, boost=1.0, width=0.8, cap=0.55, alpha=0.30,
               vkp=0.09, vsign=-1.0, tau=0.1, wheel_mm=162.0, acc_filter_s=0.10, max_lean=3.0):
    """origin/daniel-updates (src/control/PID.cpp, VelocityEstimator.h, VelocityControl.h, main.cpp) with the
    tuner values Daniel runs (2026-10-30). Defaults: MAX_BALANCE_OUTPUT 0.55, IMU_RATE_LPF_ALPHA 0.30.

    Velocity loop (vki 0): the fused velocity, wheel rpm continuously corrected toward by integrated
    gravity-compensated forward acceleration (time constant tau), sets the target pitch, capped at
    MAX_VEL_LEAN_DEG. Pitch loop: P front/back by the side of the target, plus kp * boost * tanh(error / width)
    near the target; D on the filtered pitch rate. SET_ACCEL is left at its default 0.
    """
    s = {"integ": np.zeros(n), "rate": np.zeros(n), "prev": None, "count": None, "since": np.zeros(n),
         "acc": np.zeros(n), "v": None}
    rpm_per_mps = 60000.0 / (np.pi * wheel_mm)

    def ctrl(m):
        if s["prev"] is None:
            s["prev"], s["count"] = m["pitch_deg"].copy(), m["imu_count"].copy()
        s["since"] += CONTROL_DT
        new = m["imu_count"] != s["count"]
        raw = (m["pitch_deg"] - s["prev"]) / np.maximum(s["since"], CONTROL_DT)
        s["rate"] = np.where(new, s["rate"] + alpha * (raw - s["rate"]), s["rate"])
        th = np.radians(m["pitch_deg"])   # Imu.cpp: gravity removed with the IMU's own pitch, then low-passed
        horiz = (m["accel_fwd"] + G * np.sin(th)) / np.cos(th)
        a = s["since"] / (acc_filter_s + s["since"])
        s["acc"] = np.where(new, s["acc"] + a * (horiz - s["acc"]), s["acc"])
        s["prev"] = np.where(new, m["pitch_deg"], s["prev"])
        s["since"] = np.where(new, 0.0, s["since"])
        s["count"] = m["imu_count"].copy()
        wheel = 0.5 * (m["rpm_left"] + m["rpm_right"])
        if s["v"] is None or tau <= 0:
            s["v"] = np.asarray(wheel, float).copy()
        else:
            d = np.exp(-CONTROL_DT / tau)
            s["v"] = wheel + (s["v"] - wheel) * d + s["acc"] * rpm_per_mps * tau * (1 - d)
        target = np.clip(vsign * vkp * (0.0 - s["v"]), -max_lean, max_lean) if vkp else 0.0
        err = target - m["pitch_deg"]
        s["integ"] = s["integ"] + err * CONTROL_DT
        if ki:
            s["integ"] = np.clip(s["integ"], -cap / ki, cap / ki)
        kp = np.where(err < 0, kp_front, kp_back)
        out = np.clip(kp * (err + boost * np.tanh(err / width)) + ki * s["integ"] - kd * s["rate"], -cap, cap)
        return -out  # BALANCE_MOTOR_SIGN, as firmware_pid

    return ctrl


# --- LQR -------------------------------------------------------------------------

def _expm(a):
    """Matrix exponential by scaling and squaring."""
    s = max(0, int(np.ceil(np.log2(max(np.abs(a).sum(axis=1).max(), 1e-12)))) + 1)
    a = a / 2.0 ** s
    out = term = np.eye(len(a))
    for k in range(1, 20):
        term = term @ a / k
        out = out + term
    for _ in range(s):
        out = out @ out
    return out


def _dare(a, b, q, r):
    """Discrete algebraic Riccati equation, by structure-preserving doubling."""
    g = b @ np.linalg.solve(r, b.T)
    h = q.copy()
    eye = np.eye(len(a))
    for _ in range(60):
        w = np.linalg.inv(eye + g @ h)
        a, g, h_next = a @ w @ a, g + a @ w @ g @ a.T, h + a.T @ h @ w @ a
        done = np.allclose(h_next, h, rtol=1e-13, atol=0.0)
        h = h_next
        if done:
            break
    return 0.5 * (h + h.T)


def nominal_model(rand=Randomization(), dt=CONTROL_DT, mode="speed"):
    """The nominal robot linearised about upright on a flat floor, wheels not
    slipping, with the hoverboard's stock command low-pass.

    mode "voltage": the command sets motor volts. Returns Ad (4x4) and Bd (4,)
    over [pitch, pitch rate, position, speed].
    mode "speed": the command sets a rotor speed that the hoverboard holds with
    a PI on motor current; a fifth state is the PI's integral of speed error.
    Driven by the command after the hoverboard filter, held for dt.
    """
    nom = lambda k: getattr(rand, k).nominal
    M, mt = nom("body_mass"), nom("body_mass") + nom("wheel_mass")
    L = np.hypot(nom("com_height"), nom("com_offset"))
    Jb = M * (L ** 2 + nom("body_gyration") ** 2)
    Iw, r = nom("wheel_inertia"), nom("wheel_radius")
    k = nom("back_emf")
    # Wheel torque, both motors: tu * command + tw * rotor speed + tz * PI integral.
    if mode == "voltage":
        R = nom("phase_resistance")
        volts = nom("battery_ocv") * MAX_CMD / EFERU_FULL_SCALE   # motor volts per unit command
        tu, tw, tz, ns = 2 * k / R * volts, -(2 * k * k / R + nom("viscous_friction")), 0.0, 4
    else:
        kp = nom("speed_kp")
        c_w = nom("speed_full_scale_rpm") / RPM_PER_RAD_S * MAX_CMD / EFERU_FULL_SCALE  # rad/s per unit
        tu, tw, tz, ns = (2 * k * kp * c_w, -(2 * k * kp + nom("viscous_friction")),
                          2 * k * kp * nom("speed_ki_ratio"), 5)

    # Torque tau on the wheels, -tau on the body; rotor speed w = speed / r - pitch rate.
    #   (mt + Iw/r^2) xdd + M L thdd = tau / r
    #   M L xdd + Jb thdd = -tau + M G L th
    minv = np.linalg.inv([[mt + Iw / r ** 2, M * L], [M * L, Jb]])
    e = minv @ [1.0 / r, -1.0]
    a = np.zeros((ns, ns))
    a[0, 1] = a[2, 3] = 1.0
    acc = np.zeros((2, ns))                   # rows: xdd, thdd
    acc[:, 0] = minv @ [0.0, M * G * L]
    acc[:, 1] = -e * tw
    acc[:, 3] = e * tw / r
    a[3], a[1] = acc[0], acc[1]
    b = np.zeros(ns)
    b[3], b[1] = e[0] * tu, e[1] * tu
    if ns == 5:
        a[3, 4], a[1, 4] = e[0] * tz, e[1] * tz
        a[4, 1], a[4, 3], b[4] = 1.0, -1.0 / r, c_w   # d/dt integral = target - rotor speed

    big = np.zeros((ns + 1, ns + 1))
    big[:ns, :ns], big[:ns, ns] = a * dt, b * dt
    ed = _expm(big)
    return ed[:ns, :ns], ed[:ns, ns]


def lqr(n, rand=Randomization(), mode="speed",
        q=(1 / 0.03 ** 2, 1 / 0.5 ** 2, 1 / 1.0 ** 2, 1 / 0.3 ** 2, 1 / 0.3 ** 2),
        r=1 / 0.02 ** 2, imu_noise_deg=0.1, rpm_noise=1.5, accel_noise=4.0, integral_noise=0.01,
        imu_latency=0.05):
    """LQR on the nominal model for the hoverboard `mode`, with a Kalman filter
    for the state.

    Controller state [pitch, pitch rate, position, speed, (speed mode: the
    hoverboard's PI integral,) command after the hoverboard filter, last
    command]; it chooses the change in command, so `r` penalises fast command
    changes and the last q entry large commands. The filter models the
    hoverboard low-pass, about 10 ms of command latency, `imu_latency` of IMU
    latency (in 5 ms steps), and the hoverboard's lagged, late wheel speed.
    Gains and filter are fixed: the controller does not know the robot it is on.

    ctrl(m, rpm_ref, sent): rpm_ref is the wheel rpm to drive at (CMD_VEL
    linear * MAX_WHEEL_RPM, default 0); sent is the command actually sent
    after the previous call, when something was added to this controller's.
    ctrl.reset(idx) restarts the filter for robots that were reset.
    """
    nom = lambda k: getattr(rand, k).nominal
    dt, rad = CONTROL_DT, nom("wheel_radius")
    Ad, Bd = nominal_model(rand, dt, mode)
    ns = len(Ad)
    lp = nom("cmd_lowpass")
    e_fb = np.exp(-dt / nom("fb_lag"))

    # LQR over [plant states, u_f, u_prev], input v = change in command.
    A = np.zeros((ns + 2, ns + 2))
    A[:ns, :ns], A[:ns, ns] = Ad, Bd
    A[ns, ns], A[ns, ns + 1] = 1.0 - lp, lp
    A[ns + 1, ns + 1] = 1.0
    B = np.zeros((ns + 2, 1))
    B[ns + 1, 0] = 1.0
    Q = np.diag([q[0], q[1], q[2], q[3]] + [0.0] * (ns - 4) + [0.0, q[4]])
    P = _dare(A, B, Q, np.array([[r]]))
    K = np.linalg.solve(r + B.T @ P @ B, B.T @ P @ A)[0]

    # Driving steadily at speed v: the lean, PI integral and command that hold
    # it, per m/s, as an offset to the LQR state.
    keep = [0, 1, 3] + list(range(4, ns))             # plant states but position
    unknown = [0] + list(range(4, ns))
    ad_i = Ad - np.eye(ns)
    ss = np.linalg.lstsq(np.column_stack([ad_i[np.ix_(keep, unknown)], Bd[keep]]),
                         -ad_i[keep, 3], rcond=None)[0]
    ref_per_v = np.zeros(ns + 2)
    ref_per_v[0], ref_per_v[3], ref_per_v[4:ns] = ss[0], 1.0, ss[1:-1]
    ref_per_v[ns] = ref_per_v[ns + 1] = ss[-1]

    # Kalman filter over [th, thd, xd, (PI integral,) u_f, th 5 ms ago, ..., th
    # imu_latency ago, lagged rotor speed, that 5 ms ago, that 10 ms ago], driven
    # by the previous command. Position is not observable, so it is integrated outside.
    nk = len(keep)
    nd = int(round(imu_latency / dt))
    UF, W0 = nk, nk + 1 + nd
    TH = [0] + list(range(nk + 1, nk + 1 + nd))       # pitch now, then each 5 ms older
    nz = W0 + 3
    F = np.zeros((nz, nz))
    F[np.ix_(range(nk), range(nk))] = Ad[np.ix_(keep, keep)]
    F[:nk, UF] = Bd[keep]
    F[UF, UF] = 1.0 - lp
    F[TH[1:], TH[:-1]] = 1.0
    F[W0, W0] = e_fb
    F[W0, 2], F[W0, 1] = (1 - e_fb) / rad, -(1 - e_fb)
    F[W0 + 1, W0] = F[W0 + 2, W0 + 1] = 1.0
    Gu = np.zeros(nz)
    Gu[UF] = lp
    h_imu, h_fb = np.zeros(nz), np.zeros(nz)
    h_imu[TH[-1]], h_fb[W0 + 2] = 1.0, 1.0
    r_imu = np.radians(imu_noise_deg) ** 2
    r_fb = (rpm_noise / RPM_PER_RAD_S) ** 2
    Qn = np.diag([1e-8, (accel_noise * dt) ** 2, (0.2 * accel_noise * dt) ** 2]
                 + [integral_noise ** 2] * (nk - 3) + [0.01 ** 2] + [1e-8] * nd + [1e-6, 1e-6, 1e-6])
    eye = np.eye(nz)
    P0 = np.diag([1e-3, 0.1, 0.01] + [0.1] * (nk - 3) + [1e-6] + [1e-3] * nd + [0.1, 0.1, 0.1])

    s = {"z": np.zeros((n, nz)), "P": np.tile(P0, (n, 1, 1)), "x": np.zeros(n), "u": np.zeros(n),
         "imu": None, "fb": None, "fresh": np.ones(n, dtype=bool)}

    def update(mask, h, y, rn):
        if not mask.any():
            return
        z, P = s["z"], s["P"]
        ph = P @ h                                    # (n, 9)
        gain = ph / (ph @ h + rn)[:, None]
        innov = np.where(mask, y - z @ h, 0.0)
        s["z"] = z + gain * innov[:, None]
        ikh = eye - gain[:, :, None] * h[None, None, :]
        joseph = ikh @ P @ ikh.transpose(0, 2, 1) + rn * gain[:, :, None] * gain[:, None, :]
        s["P"] = np.where(mask[:, None, None], joseph, P)

    def ctrl(m, rpm_ref=None, sent=None):
        pitch = m["pitch_deg"] / DEG
        w = 0.5 * (m["rpm_left"] + m["rpm_right"]) / RPM_PER_RAD_S
        if s["imu"] is None:
            s["imu"], s["fb"] = m["imu_count"].copy(), m["fb_count"].copy()
        f = s["fresh"]
        if f.any():
            z0 = np.zeros((f.sum(), nz))
            z0[:, TH] = pitch[f, None]
            s["z"][f], s["P"][f] = z0, P0
            s["x"][f] = s["u"][f] = 0.0
            s["imu"][f], s["fb"][f] = m["imu_count"][f], m["fb_count"][f]
            f[:] = False
        update(m["imu_count"] != s["imu"], h_imu, pitch, r_imu)
        update(m["fb_count"] != s["fb"], h_fb, w, r_fb)
        s["imu"], s["fb"] = m["imu_count"].copy(), m["fb_count"].copy()

        v_ref = 0.0 if rpm_ref is None else rpm_ref / RPM_PER_RAD_S * rad
        z, u_prev = s["z"], s["u"]
        state = np.column_stack([z[:, 0], z[:, 1], s["x"], z[:, 2:nk], z[:, UF], u_prev])
        u = np.clip(u_prev - (state - np.multiply.outer(v_ref, ref_per_v)) @ K, -1.0, 1.0)

        s["x"] = s["x"] + dt * (z[:, 2] - v_ref)
        s["z"] = z @ F.T + (u_prev if sent is None else sent)[:, None] * Gu
        P = F @ s["P"] @ F.T + Qn
        s["P"] = 0.5 * (P + P.transpose(0, 2, 1))
        s["u"] = u
        return u

    def reset(idx):
        s["fresh"][idx] = True

    ctrl.gain = K
    ctrl.reset = reset
    return ctrl


CONTROLLERS = {
    "none": ("No control", no_control),
    "pid": ("Cascaded PID", cascaded_pid),
    "lqr": ("LQR", lqr),
}
