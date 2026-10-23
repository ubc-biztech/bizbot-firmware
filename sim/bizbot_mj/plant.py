"""Everything outside the ESP32, with MuJoCo doing the rigid-body and contact
physics: n robots in one MuJoCo model, each on its own floor.

Each robot is a frame on a free joint at the axle, an upper body on a pitch
hinge above it (locked by an equality constraint on stiff robots, a damped
spring on flexible ones), and two cylinder wheels on hinges. Wheels touch only
their own robot's floor plane through explicit contact pairs, so robots never
interact. Mass, geometry, friction, tyre compliance and slope are rewritten in
the model at every reset.

Hub motors are MuJoCo actuators on the wheel hinges: the hoverboard's speed
loop proportional term is actuator damping (integrated implicitly), the rest
of its command is the actuator input, and current and battery-voltage limits
are the actuator's force range, updated every physics step. Drivetrain
stiction is MuJoCo joint friction loss.

The hoverboard command filter, the IMU and the wheel feedback are delays and
filters on ring buffers at physics resolution, as in sim/bizbot_sim.
"""

import mujoco
import numpy as np

from .params import sample

G = 9.81
MG = 9.80665e-3             # 1 mg, the RVC acceleration resolution
DEG = 180.0 / np.pi
RPM_PER_RAD_S = 60.0 / (2.0 * np.pi)
EFERU_LOOP_S = 0.005        # EFeru main loop period
EFERU_FULL_SCALE = 1000.0   # EFeru command range is +-1000
BATT_FILTER_S = 0.5         # hoverboard battery voltage filter
ROLL_EPS = 0.5              # rad/s; width of the rolling resistance transition
WHEEL_HALF_WIDTH = 0.035    # m
NQ, NV = 10, 9              # per robot: free joint (7/6), flex, left, right

CHANGEABLE = ("slope_deg", "mu", "com_offset", "body_mass")


def _xml(n, dt):
    bodies, pairs, eqs, acts, sens = [], [], [], [], []
    for i in range(n):
        bodies.append(f"""
  <geom name="floor{i}" type="plane" size="0 0 1"/>
  <body name="base{i}" pos="0 0 0.0825">
    <freejoint/>
    <inertial pos="0 0 0.02" mass="4" diaginertia="0.05 0.05 0.05"/>
    <site name="imu_b{i}" pos="0 0 0.1"/>
    <body name="upper{i}" pos="0 0 0.1">
      <joint name="flex{i}" type="hinge" axis="0 1 0"/>
      <inertial pos="0 0 0.35" mass="16" diaginertia="1 1 0.3"/>
      <geom name="shell{i}" type="box" size="0.08 0.15 0.3" pos="0 0 0.3" mass="0"/>
      <site name="imu_u{i}" pos="0 0 0.15"/>
    </body>
    <body name="wl{i}" pos="0 0.24 0">
      <joint name="jl{i}" type="hinge" axis="0 1 0"/>
      <inertial pos="0 0 0" mass="3.5" diaginertia="0.01 0.01 0.01"/>
      <geom name="gl{i}" type="cylinder" size="0.0825 {WHEEL_HALF_WIDTH}" zaxis="0 1 0" mass="0"/>
    </body>
    <body name="wr{i}" pos="0 -0.24 0">
      <joint name="jr{i}" type="hinge" axis="0 1 0"/>
      <inertial pos="0 0 0" mass="3.5" diaginertia="0.01 0.01 0.01"/>
      <geom name="gr{i}" type="cylinder" size="0.0825 {WHEEL_HALF_WIDTH}" zaxis="0 1 0" mass="0"/>
    </body>
  </body>""")
        pairs.append(f'<pair name="pl{i}" geom1="gl{i}" geom2="floor{i}" condim="3"/>'
                     f'<pair name="pr{i}" geom1="gr{i}" geom2="floor{i}" condim="3"/>')
        eqs.append(f'<joint name="lock{i}" joint1="flex{i}" solref="{2 * dt} 1" solimp="0.95 0.99 0.001"/>')
        for side in "lr":
            acts.append(f'<general name="m{side}{i}" joint="j{side}{i}" gainprm="1" biastype="affine" '
                        f'biasprm="0 0 0" forcelimited="true" forcerange="-15 15"/>')
        sens.append(f'<accelerometer site="imu_b{i}"/><accelerometer site="imu_u{i}"/>')
    return f"""<mujoco model="bizbot">
<option timestep="{dt}" integrator="implicitfast" cone="pyramidal"/>
<default><geom contype="0" conaffinity="0"/></default>
<worldbody>{''.join(bodies)}
</worldbody>
<contact>{''.join(pairs)}</contact>
<equality>{''.join(eqs)}</equality>
<actuator>{''.join(acts)}</actuator>
<sensor>{''.join(sens)}</sensor>
</mujoco>"""


def _quat_y(angle):
    """Quaternions (w, x, y, z) for rotations about +y; positive tips +z toward +x."""
    a = 0.5 * np.asarray(angle)
    q = np.zeros(a.shape + (4,))
    q[..., 0], q[..., 2] = np.cos(a), np.sin(a)
    return q


class Plant:
    def __init__(self, n, rand, dt, rng, scale=1.0):
        self.n, self.rand, self.dt, self.rng, self.scale = n, rand, dt, rng, scale
        self.m = mujoco.MjModel.from_xml_string(_xml(n, dt))
        self.d = mujoco.MjData(self.m)
        m = self.m
        nid = lambda kind, name: mujoco.mj_name2id(m, kind, name)
        B, J, Gm, A = (mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_JOINT,
                       mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_ACTUATOR)
        ids = lambda kind, fmt: np.array([nid(kind, fmt.format(i)) for i in range(n)])
        self.b_base, self.b_up = ids(B, "base{}"), ids(B, "upper{}")
        self.b_wl, self.b_wr = ids(B, "wl{}"), ids(B, "wr{}")
        self.j_flex = ids(J, "flex{}")
        self.g_floor, self.g_wl, self.g_wr = ids(Gm, "floor{}"), ids(Gm, "gl{}"), ids(Gm, "gr{}")
        self.g_shell = ids(Gm, "shell{}")
        self.a_l, self.a_r = ids(A, "ml{}"), ids(A, "mr{}")
        self.s_imu_b = ids(mujoco.mjtObj.mjOBJ_SITE, "imu_b{}")
        self.s_imu_u = ids(mujoco.mjtObj.mjOBJ_SITE, "imu_u{}")
        self.p_l = ids(mujoco.mjtObj.mjOBJ_PAIR, "pl{}")
        self.p_r = ids(mujoco.mjtObj.mjOBJ_PAIR, "pr{}")
        self.eq = ids(mujoco.mjtObj.mjOBJ_EQUALITY, "lock{}")
        assert (np.diff(self.a_l) == 2).all() and (self.a_r == self.a_l + 1).all()
        assert m.nq == NQ * n and m.nv == NV * n and m.nsensordata == 6 * n
        assert (m.jnt_qposadr[self.j_flex] == NQ * np.arange(n) + 7).all()
        self.dof_l = m.jnt_dofadr[ids(J, "jl{}")]
        self.dof_r = m.jnt_dofadr[ids(J, "jr{}")]
        self.dof_flex = m.jnt_dofadr[self.j_flex]

        max_latency = max(rand.imu_latency.hi + rand.imu_jitter.hi, rand.cmd_latency.hi,
                          rand.fb_latency.hi)
        self.H = int(np.ceil(max_latency / dt)) + 2
        self.ptr = 0
        self.p = sample(rand, rng, n, scale)
        for k in ("k_m", "r_m", "imax_m", "fb_gain"):
            self.p[k] = np.ones((2, n))
        for k in ("spd_full", "spd_ki", "kv", "rate", "slope", "imu_up", "w_load", "L_push",
                  "gyration_real"):
            self.p[k] = np.zeros(n)
        for k in ("cmd_delay", "imu_delay", "fb_delay"):
            self.p[k] = np.zeros(n, dtype=np.int64)

        for k in ("t u_rl u_f i_dc batt_lp batt_meas imu_err imu_lp rpm_lag imu_pitch imu_accel "
                  "imu_next fb_next eferu_next push_F push_until push_h push_next change_next "
                  "fwd_acc").split():
            setattr(self, k, np.zeros(n))
        self.v_batt = np.zeros(n)
        self.spd_int = np.zeros((2, n))
        self.fb_rpm = np.zeros((2, n))
        self.imu_count = np.zeros(n, dtype=np.int64)
        self.fb_count = np.zeros(n, dtype=np.int64)
        self.hist_cmd = np.zeros((self.H, n))
        self.hist_pitch = np.zeros((self.H, n))
        self.hist_accel = np.zeros((self.H, n))
        self.hist_rpm = np.zeros((self.H, n))
        self.reset(np.arange(n), np.zeros(n), np.zeros(n), np.zeros(n))

    # --- views ---------------------------------------------------------------

    @property
    def qpos(self):
        return self.d.qpos.reshape(self.n, NQ)

    @property
    def qvel(self):
        return self.d.qvel.reshape(self.n, NV)

    def base_pitch(self):
        """Frame pitch from vertical (rad, + forward) and the frame's heading in the floor plane."""
        q = self.qpos[:, 3:7]
        w, x, y, z = q.T
        zx = 2.0 * (x * z + w * y)           # body z axis, world x component
        zz = 1.0 - 2.0 * (x * x + y * y)
        return np.arctan2(zx, zz)

    def speed(self):
        """Axle speed along the robot's heading (m/s)."""
        q = self.qpos[:, 3:7]
        w, x, y, z = q.T
        hx, hy = 1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z)   # body x axis in world
        norm = np.maximum(np.hypot(hx, hy), 1e-9)
        v = self.qvel
        return (v[:, 0] * hx + v[:, 1] * hy) / norm

    # --- setup -----------------------------------------------------------------

    def _write_body(self, idx):
        """Masses, geometry and contacts for robots idx from self.p."""
        m, p = self.m, self.p
        M, h, d = p["body_mass"][idx], p["com_height"][idx], p["com_offset"][idx]
        share = p["base_share"][idx]
        hb = np.minimum(0.04, 0.3 * h)                   # frame COM height
        hj = p["flex_height"][idx] * h                   # flex hinge height
        mb, mu_ = share * M, (1.0 - share) * M
        # Upper-body COM such that the whole body's COM is at (d, h).
        hu = (M * h - mb * hb) / mu_
        du = M * d / mu_
        # Pitch inertia about the whole COM is M * gyration^2; give the upper
        # body what is left after the frame's point mass and parallel axes.
        Jtot = M * p["body_gyration"][idx] ** 2
        Ju = np.maximum(Jtot - mb * ((h - hb) ** 2 + d ** 2) - mu_ * ((hu - h) ** 2 + (du - d) ** 2),
                        0.05 * Jtot)
        # Pitch inertia the robot really has about its COM (Ju may have been floored).
        Jreal = (mb * ((h - hb) ** 2 + d ** 2) + mb * 0.01 + 1e-3 + Ju
                 + mu_ * ((hu - h) ** 2 + (du - d) ** 2))
        p["gyration_real"][idx] = np.sqrt(Jreal / M)
        m.body_mass[self.b_base[idx]] = mb
        m.body_ipos[self.b_base[idx]] = np.stack([np.zeros_like(hb), np.zeros_like(hb), hb], 1)
        m.body_inertia[self.b_base[idx]] = (mb * 0.01)[:, None] * np.ones(3) + 1e-3
        m.body_pos[self.b_up[idx]] = np.stack([np.zeros_like(hj), np.zeros_like(hj), hj], 1)
        m.body_mass[self.b_up[idx]] = mu_
        m.body_ipos[self.b_up[idx]] = np.stack([du, np.zeros_like(du), hu - hj], 1)
        side = mu_ * 0.04 + 0.2 * Ju
        m.body_inertia[self.b_up[idx]] = np.stack([Ju + side, Ju, side], 1)  # roll, pitch, yaw
        m.geom_pos[self.g_shell[idx], 2] = np.maximum(hu - hj, 0.05)

        # Flex: the hinge is held by a soft equality constraint, which MuJoCo
        # solves implicitly (a joint spring is explicit and blows up on light
        # frames). Stiff robots get the stiffest the timestep allows (about
        # 40 Hz, critically damped); flexible ones a reference acceleration of
        # -wn^2 angle - 2 zeta wn rate.
        wn = 2.0 * np.pi * p["flex_hz"][idx]
        flexible = p["flex"][idx].astype(bool)
        m.eq_solref[self.eq[idx], 0] = np.where(flexible, -wn ** 2, 2.0 * self.dt)
        m.eq_solref[self.eq[idx], 1] = np.where(flexible, -2.0 * p["flex_zeta"][idx] * wn, 1.0)

        # IMU on the upper body when it sits above the flex hinge.
        ih = p["imu_height"][idx]
        p["imu_up"][idx] = ih >= hj
        m.site_pos[self.s_imu_b[idx], 2] = ih
        m.site_pos[self.s_imu_u[idx], 2] = ih - hj

        # Wheels.
        r = p["wheel_radius"][idx]
        mw, Iw = 0.5 * p["wheel_mass"][idx], 0.5 * p["wheel_inertia"][idx]
        side_I = 0.5 * Iw + mw * WHEEL_HALF_WIDTH ** 2 / 3.0
        tw = 0.5 * p["track"][idx]
        for b, g, sgn in ((self.b_wl, self.g_wl, 1.0), (self.b_wr, self.g_wr, -1.0)):
            m.body_mass[b[idx]] = mw
            m.body_inertia[b[idx]] = np.stack([side_I, Iw, side_I], 1)
            m.body_pos[b[idx], 1] = sgn * tw
            m.geom_size[g[idx], 0] = r
            m.geom_rbound[g[idx]] = np.hypot(r, WHEEL_HALF_WIDTH)
        m.dof_frictionloss[self.dof_l[idx]] = 0.5 * p["coulomb_friction"][idx]
        m.dof_frictionloss[self.dof_r[idx]] = 0.5 * p["coulomb_friction"][idx]
        m.dof_damping[self.dof_l[idx]] = 0.5 * p["viscous_friction"][idx]
        m.dof_damping[self.dof_r[idx]] = 0.5 * p["viscous_friction"][idx]

        # Floor.
        self._write_floor(idx)
        p["w_load"][idx] = 0.5 * (M + p["wheel_mass"][idx]) * G
        p["L_push"][idx] = h

    def _write_floor(self, idx):
        m, p = self.m, self.p
        p["slope"][idx] = np.radians(p["slope_deg"][idx])
        m.geom_quat[self.g_floor[idx]] = _quat_y(-p["slope"][idx])
        for pr in (self.p_l, self.p_r):
            m.pair_friction[pr[idx], 0] = p["mu"][idx]
            m.pair_friction[pr[idx], 1] = p["mu"][idx]
            m.pair_solref[pr[idx], 0] = p["contact_tc"][idx]
            m.pair_solref[pr[idx], 1] = p["contact_damp"][idx]

    def _write_motors(self, idx):
        p, m, rng, k = self.p, self.m, self.rng, len(idx)
        spread = lambda: 1.0 + p["motor_mismatch"][idx] * rng.uniform(-1, 1, (2, k))
        p["k_m"][:, idx] = p["back_emf"][idx] * spread()
        p["r_m"][:, idx] = p["phase_resistance"][idx] * spread()
        p["imax_m"][:, idx] = p["current_limit"][idx] * spread()
        p["fb_gain"][:, idx] = 1.0 + p["fb_gain_error"][idx] * rng.uniform(-1, 1, (2, k))
        p["rate"][idx] = np.where(p["cmd_rate_limit_off"][idx], np.inf, p["cmd_rate_limit"][idx])
        p["spd_full"][idx] = p["speed_full_scale_rpm"][idx] / RPM_PER_RAD_S
        p["spd_ki"][idx] = p["speed_kp"][idx] * p["speed_ki_ratio"][idx]
        # Speed-loop proportional term as implicit actuator damping, per motor.
        m.actuator_biasprm[self.a_l[idx], 2] = -p["k_m"][0, idx] * p["speed_kp"][idx]
        m.actuator_biasprm[self.a_r[idx], 2] = -p["k_m"][1, idx] * p["speed_kp"][idx]
        dt = self.dt
        p["cmd_delay"][idx] = np.rint(p["cmd_latency"][idx] / dt)
        p["imu_delay"][idx] = np.rint(p["imu_latency"][idx] / dt)
        p["fb_delay"][idx] = np.rint(p["fb_latency"][idx] / dt)

    def _exp(self, rate):
        draw = self.rng.exponential(1.0, rate.shape)
        return np.where(rate > 0, draw / np.maximum(rate, 1e-12), np.inf)

    def reset(self, idx, pitch, rate, speed):
        """New robots at idx: pitch from vertical (rad), pitch rate (rad/s), speed (m/s)."""
        idx = np.asarray(idx)
        if idx.size == 0:
            return
        rng, p, k = self.rng, self.p, len(idx)
        for key, v in sample(self.rand, rng, k, self.scale).items():
            p[key][idx] = v
        self.m.geom_pos[self.g_floor[idx]] = 0.0
        self._write_body(idx)
        self._write_motors(idx)

        r, s = p["wheel_radius"][idx], p["slope"][idx]
        qp, qv = self.qpos, self.qvel
        qp[idx] = 0.0
        qv[idx] = 0.0
        # Contact point at the origin, axle r along the floor normal.
        qp[idx, 0], qp[idx, 2] = -r * np.sin(s), r * np.cos(s)
        qp[idx, 3:7] = _quat_y(pitch)
        qv[idx, 0], qv[idx, 2] = speed * np.cos(s), speed * np.sin(s)
        qv[idx, 4] = rate
        qv[idx, 7] = qv[idx, 8] = speed / r - rate
        self.d.xfrc_applied[self.b_up[idx]] = 0.0
        self.d.xfrc_applied[self.b_wl[idx]] = 0.0
        self.d.xfrc_applied[self.b_wr[idx]] = 0.0
        self.d.qacc_warmstart.reshape(self.n, NV)[idx] = 0.0

        self.t[idx] = 0.0
        self.u_rl[idx] = self.u_f[idx] = self.i_dc[idx] = 0.0
        self.spd_int[:, idx] = 0.0
        self.v_batt[idx] = self.batt_lp[idx] = self.batt_meas[idx] = p["battery_ocv"][idx]
        self.imu_err[idx] = 0.0
        self.imu_lp[idx] = pitch
        self.fwd_acc[idx] = 0.0
        self.rpm_lag[idx] = (speed / r) * RPM_PER_RAD_S - rate * RPM_PER_RAD_S
        self.push_F[idx] = self.push_until[idx] = 0.0
        self.push_next[idx] = self._exp(p["push_rate"][idx])
        self.change_next[idx] = self._exp(p["change_rate"][idx])
        self.eferu_next[idx] = rng.uniform(0, EFERU_LOOP_S, k)
        self.imu_next[idx] = rng.uniform(0, p["imu_period"][idx])
        self.fb_next[idx] = rng.uniform(0, p["fb_period"][idx])

        self.hist_cmd[:, idx] = 0.0
        self.hist_pitch[:, idx] = pitch
        self.hist_accel[:, idx] = -G * np.sin(pitch)
        self.hist_rpm[:, idx] = self.rpm_lag[idx]
        self.imu_pitch[idx] = np.round((pitch * DEG + p["imu_mount_deg"][idx]) * 100.0) / 100.0
        self.imu_accel[idx] = -G * np.sin(pitch)
        self.fb_rpm[:, idx] = np.round(self.rpm_lag[idx] * p["fb_gain"][:, idx])
        self.imu_count[idx] = self.fb_count[idx] = 0

    # --- stepping --------------------------------------------------------------

    def advance(self, cmd, substeps):
        """Hold the ESP32 command (hoverboard units) for `substeps` physics steps."""
        for _ in range(substeps):
            self._substep(cmd)

    def _substep(self, cmd):
        p, dt, H, rng, ptr, m, d = self.p, self.dt, self.H, self.rng, self.ptr, self.m, self.d
        self.hist_cmd[ptr] = cmd

        # EFeru main loop: deadband, rate limiter, low-pass on the delayed command.
        due = np.nonzero(self.t >= self.eferu_next)[0]
        if due.size:
            c_in = self.hist_cmd[(ptr - p["cmd_delay"][due]) % H, due]
            c_in = np.where(np.abs(c_in) <= p["cmd_deadband"][due], 0.0, c_in)
            lim = p["rate"][due]
            self.u_rl[due] += np.clip(c_in - self.u_rl[due], -lim, lim)
            self.u_f[due] += p["cmd_lowpass"][due] * (self.u_rl[due] - self.u_f[due])
            self.eferu_next[due] += EFERU_LOOP_S

        # Hub motors in speed mode: i = kp (target - w) + ki * integral, the
        # kp * w part being the actuator's implicit damping.
        qv = self.qvel
        w = np.stack([qv[:, 7], qv[:, 8]])                       # rotor speed relative to the frame
        target = np.clip(self.u_f / EFERU_FULL_SCALE, -1.0, 1.0) * p["spd_full"]
        k, R, imax = p["k_m"], p["r_m"], p["imax_m"]
        kp, ki = p["speed_kp"], p["spd_ki"]
        self.spd_int += (target - w) * dt
        wind = imax / np.maximum(ki, 1e-9)
        np.clip(self.spd_int, -wind, wind, out=self.spd_int)
        ctrl = k * (kp * target + ki * self.spd_int)
        vb = self.v_batt
        hi = k * np.minimum(imax, (vb - k * w) / R)
        lo = k * np.maximum(-imax, (-vb - k * w) / R)
        hi = np.maximum(hi, lo + 1e-6)
        d.ctrl.reshape(self.n, 2)[:] = ctrl.T
        fr = m.actuator_forcerange.reshape(self.n, 2, 2)
        fr[:, :, 0] = lo.T
        fr[:, :, 1] = hi.T

        # Pushes on the upper body; rolling resistance on the wheels.
        th = self.base_pitch()
        pushing = self.t < self.push_until
        if pushing.any() or (self.push_F != 0).any():
            F = np.where(pushing, self.push_F, 0.0)
            arm = self.push_h - (m.body_ipos[self.b_up, 2] + m.body_pos[self.b_up, 2])
            xf = d.xfrc_applied
            xf[self.b_up, 0] = F
            xf[self.b_up, 4] = F * arm * np.cos(th)
            self.push_F = np.where(pushing, self.push_F, 0.0)
        w_abs_l = w[0] + qv[:, 4]
        w_abs_r = w[1] + qv[:, 4]
        rr = p["rolling_resistance"] * p["w_load"] * p["wheel_radius"]
        d.xfrc_applied[self.b_wl, 4] = -rr * np.tanh(w_abs_l / ROLL_EPS)
        d.xfrc_applied[self.b_wr, 4] = -rr * np.tanh(w_abs_r / ROLL_EPS)

        mujoco.mj_step(m, d)

        # Battery from the motor currents that flowed.
        force = d.actuator_force.reshape(self.n, 2).T
        i = force / k
        self.i_dc = ((R * i + k * w) * i).sum(axis=0) / vb
        self.v_batt = np.maximum(p["battery_ocv"] - p["battery_resistance"] * self.i_dc,
                                 0.5 * p["battery_ocv"])
        self.batt_lp += dt / BATT_FILTER_S * (self.v_batt - self.batt_lp)

        # What the sensors would see now; delivered later through the histories.
        qv = self.qvel
        th = self.base_pitch()
        flex = self.qpos[:, 7]
        th_imu = np.where(p["imu_up"] > 0, th + flex, th)
        acc_w = d.qacc.reshape(self.n, NV)[:, 0]
        self.fwd_acc = acc_w
        self.imu_err += dt / p["imu_fusion_s"] * (p["imu_accel_tilt"] * acc_w / G - self.imu_err)
        a_lp = np.where(p["imu_filter"] > 0, dt / np.maximum(p["imu_filter"], dt), 1.0)
        self.imu_lp += np.minimum(a_lp, 1.0) * (th_imu - self.imu_lp)
        sd = d.sensordata.reshape(self.n, 2, 3)
        acc_imu = np.where(p["imu_up"] > 0, sd[:, 1, 0], sd[:, 0, 0])
        self.hist_pitch[ptr] = self.imu_lp + self.imu_err
        self.hist_accel[ptr] = acc_imu
        wm = 0.5 * (qv[:, 7] + qv[:, 8]) * RPM_PER_RAD_S
        self.rpm_lag += dt / p["fb_lag"] * (wm - self.rpm_lag)
        self.hist_rpm[ptr] = self.rpm_lag
        self.t += dt

        # IMU packets: pitch in 0.01 deg, acceleration in 1 mg.
        due = np.nonzero(self.t >= self.imu_next)[0]
        if due.size:
            self.imu_next[due] += p["imu_period"][due]
            due = due[rng.random(due.size) >= p["imu_drop_prob"][due]]
            age = p["imu_delay"][due] + np.rint(rng.uniform(0.0, p["imu_jitter"][due]) / dt).astype(np.int64)
            rows = (ptr - age) % H
            pitch = (self.hist_pitch[rows, due] * DEG + p["imu_mount_deg"][due]
                     + p["imu_noise_deg"][due] * rng.standard_normal(due.size))
            self.imu_pitch[due] = np.round(pitch * 100.0) / 100.0
            acc = self.hist_accel[rows, due] + p["accel_noise"][due] * rng.standard_normal(due.size)
            self.imu_accel[due] = np.round(acc / MG) * MG
            self.imu_count[due] += 1

        # Hoverboard feedback frames: integer rpm per motor, filtered battery voltage.
        due = np.nonzero(self.t >= self.fb_next)[0]
        if due.size:
            self.fb_next[due] += p["fb_period"][due]
            due = due[rng.random(due.size) >= p["fb_drop_prob"][due]]
            rpm = self.hist_rpm[(ptr - p["fb_delay"][due]) % H, due] * p["fb_gain"][:, due]
            rpm[np.abs(rpm) < p["fb_min_rpm"][due]] = 0.0
            self.fb_rpm[:, due] = np.round(rpm)
            self.batt_meas[due] = np.round(self.batt_lp[due] * 100.0) / 100.0
            self.fb_count[due] += 1

        # Pushes at a random height on the body.
        due = np.nonzero(self.t >= self.push_next)[0]
        if due.size:
            k_ = due.size
            self.push_F[due] = (rng.choice([-1.0, 1.0], k_) * rng.uniform(0, 1, k_)
                                * p["push_force_max"][due])
            self.push_until[due] = self.t[due] + rng.uniform(0.05, 0.4, k_)
            self.push_h[due] = rng.uniform(0.3, 1.2, k_) * p["L_push"][due]
            self.push_next[due] = self.t[due] + self._exp(p["push_rate"][due])

        # Sudden changes: new slope (about the contact point), floor patch, or payload.
        due = np.nonzero(self.t >= self.change_next)[0]
        if due.size:
            new = sample(self.rand, rng, due.size, self.scale)
            for key in CHANGEABLE:
                p[key][due] = new[key]
            r = p["wheel_radius"][due]
            old_n = m.geom_quat[self.g_floor[due]]
            # Floor normal before the change (rotation about y by -slope_old).
            s_old = -2.0 * np.arctan2(old_n[:, 2], old_n[:, 0])
            contact = self.qpos[due, 0:3].copy()
            contact[:, 0] += r * np.sin(s_old)
            contact[:, 2] -= r * np.cos(s_old)
            m.geom_pos[self.g_floor[due]] = contact
            self._write_body(due)
            self.change_next[due] = self.t[due] + self._exp(p["change_rate"][due])

        self.ptr = (ptr + 1) % H

    # --- readout ---------------------------------------------------------------

    def measurements(self):
        """What the ESP32 has right now (latest packets, already delayed)."""
        return {
            "pitch_deg": self.imu_pitch, "accel_fwd": self.imu_accel, "imu_count": self.imu_count,
            "rpm_left": self.fb_rpm[0], "rpm_right": self.fb_rpm[1], "fb_count": self.fb_count,
            "battery_v": self.batt_meas,
        }

    def truth(self):
        """True state, for rewards and plots. Never available on the robot."""
        qv = self.qvel
        r = self.p["wheel_radius"]
        v = self.speed()
        return {
            "pitch": self.base_pitch(), "pitch_rate": qv[:, 4].copy(), "speed": v,
            "position": self.qpos[:, 0].copy(), "flex": self.qpos[:, 7].copy(),
            "flex_rate": qv[:, 6].copy(), "wheel_rate": 0.5 * (qv[:, 7] + qv[:, 8]),
            "slip": v - r * (0.5 * (qv[:, 7] + qv[:, 8]) + qv[:, 4]),
            "slope": self.p["slope"].copy(), "push": np.where(self.t < self.push_until, self.push_F, 0.0),
            "command_at_motor": self.u_f.copy(), "battery_v": self.v_batt.copy(),
            "battery_a": self.i_dc.copy(), "accel": self.fwd_acc.copy(),
        }
