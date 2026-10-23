"""Everything outside the ESP32: body, wheels, floor, hoverboard, battery, and
the IMU and wheel feedback as the ESP32 receives them.

Planar model. Generalised coordinates are axle position x along the floor,
body pitch th from the floor normal, and wheel spin phi. The floor rises by
`slope` ahead. The body COM sits com_height up the body axis and com_offset
forward of it. Wheels roll without slipping until the friction they need
exceeds mu_s * N, then slip against mu_k * N. Friction is solved at the
velocity level every substep, so the contact is never stiff.

Delays are ring buffers at physics resolution: each link reads its signal
`latency` seconds in the past.
"""

import numpy as np

from .params import sample

G = 9.81
MG = 9.80665e-3             # 1 mg, the RVC acceleration resolution
DEG = 180.0 / np.pi
RPM_PER_RAD_S = 60.0 / (2.0 * np.pi)
VOLTAGE, SPEED, TORQUE = 0, 1, 2
EFERU_LOOP_S = 0.005        # EFeru main loop period
EFERU_FULL_SCALE = 1000.0   # EFeru command range is +-1000
SLIP_EPS = 1e-3             # m/s; below this the static coefficient applies
COULOMB_EPS = 0.2           # rad/s; width of the drivetrain friction transition
ROLL_EPS = 0.5              # rad/s; width of the rolling resistance transition
BATT_FILTER_S = 0.5         # hoverboard battery voltage filter

CHANGEABLE = ("slope_deg", "mu_kinetic", "mu_static_ratio", "com_offset", "body_mass")


class Plant:
    def __init__(self, n, rand, dt, rng, scale=1.0):
        self.n, self.rand, self.dt, self.rng, self.scale = n, rand, dt, rng, scale
        max_latency = max(rand.imu_latency.hi + rand.imu_jitter.hi, rand.cmd_latency.hi,
                          rand.fb_latency.hi)
        self.H = int(np.ceil(max_latency / dt)) + 2
        self.ptr = 0

        p = self.p = sample(rand, rng, n, scale)
        for k in ("k_m", "r_m", "imax_m", "fb_gain"):
            p[k] = np.zeros((2, n))
        for k in ("L", "phi0", "mt", "Jb", "slope", "mu_s", "rate", "spd_full", "spd_ki"):
            p[k] = np.zeros(n)
        for k in ("cmd_delay", "imu_delay", "fb_delay"):
            p[k] = np.zeros(n, dtype=np.int64)

        for k in ("t x xd th thd phi phid psidd u_rl u_f spd_int i_dc v_batt batt_lp batt_meas "
                  "imu_err rpm_lag imu_pitch imu_accel imu_next fb_next eferu_next "
                  "push_F push_until push_h push_next change_next").split():
            setattr(self, k, np.zeros(n))
        self.fb_rpm = np.zeros((2, n))
        self.imu_count = np.zeros(n, dtype=np.int64)
        self.fb_count = np.zeros(n, dtype=np.int64)
        self.hist_cmd = np.zeros((self.H, n))
        self.hist_pitch = np.zeros((self.H, n))
        self.hist_accel = np.zeros((self.H, n))
        self.hist_rpm = np.zeros((self.H, n))

    # --- setup -----------------------------------------------------------------

    def _derive(self, idx):
        p = self.p
        M, h, d = p["body_mass"][idx], p["com_height"][idx], p["com_offset"][idx]
        p["L"][idx] = np.hypot(h, d)
        p["phi0"][idx] = np.arctan2(d, h)
        p["mt"][idx] = M + p["wheel_mass"][idx]
        p["Jb"][idx] = M * (p["L"][idx] ** 2 + p["body_gyration"][idx] ** 2)
        p["slope"][idx] = np.radians(p["slope_deg"][idx])
        p["mu_s"][idx] = p["mu_kinetic"][idx] * p["mu_static_ratio"][idx]

    def _exp(self, rate):
        """Time to the next event of a Poisson process; inf when rate is 0."""
        draw = self.rng.exponential(1.0, rate.shape)
        return np.where(rate > 0, draw / np.maximum(rate, 1e-12), np.inf)

    def reset(self, idx, pitch, rate, speed):
        """New robots at idx. pitch (rad, from vertical), rate (rad/s), speed (m/s)."""
        rng, p, m, dt = self.rng, self.p, len(idx), self.dt
        for k, v in sample(self.rand, rng, m, self.scale).items():
            p[k][idx] = v
        spread = lambda: 1.0 + p["motor_mismatch"][idx] * rng.uniform(-1, 1, (2, m))
        p["k_m"][:, idx] = p["back_emf"][idx] * spread()
        p["r_m"][:, idx] = p["phase_resistance"][idx] * spread()
        p["imax_m"][:, idx] = p["current_limit"][idx] * spread()
        p["fb_gain"][:, idx] = 1.0 + p["fb_gain_error"][idx] * rng.uniform(-1, 1, (2, m))
        p["rate"][idx] = np.where(p["rate_limit_off"][idx], np.inf, p["cmd_rate_limit"][idx])
        p["spd_full"][idx] = p["speed_full_scale_rpm"][idx] / RPM_PER_RAD_S
        p["spd_ki"][idx] = p["speed_kp"][idx] * p["speed_ki_ratio"][idx]
        p["cmd_delay"][idx] = np.rint(p["cmd_latency"][idx] / dt)
        p["imu_delay"][idx] = np.rint(p["imu_latency"][idx] / dt)
        p["fb_delay"][idx] = np.rint(p["fb_latency"][idx] / dt)
        self._derive(idx)

        r = p["wheel_radius"][idx]
        self.t[idx] = 0.0
        self.x[idx] = 0.0
        self.th[idx] = p["slope"][idx] + pitch
        self.thd[idx] = rate
        self.xd[idx] = speed
        self.phid[idx] = speed / r
        self.phi[idx] = 0.0
        self.psidd[idx] = 0.0
        self.u_rl[idx] = self.u_f[idx] = self.spd_int[idx] = self.i_dc[idx] = 0.0
        self.v_batt[idx] = self.batt_lp[idx] = self.batt_meas[idx] = p["battery_ocv"][idx]
        self.imu_err[idx] = 0.0
        self.rpm_lag[idx] = (self.phid[idx] - self.thd[idx]) * RPM_PER_RAD_S
        self.push_F[idx] = self.push_until[idx] = 0.0
        self.push_next[idx] = self._exp(p["push_rate"][idx])
        self.change_next[idx] = self._exp(p["change_rate"][idx])
        self.eferu_next[idx] = rng.uniform(0, EFERU_LOOP_S, m)
        self.imu_next[idx] = rng.uniform(0, p["imu_period"][idx])
        self.fb_next[idx] = rng.uniform(0, p["fb_period"][idx])

        # Histories as if the robot had been held in this state.
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
        p, dt, H, rng, ptr = self.p, self.dt, self.H, self.rng, self.ptr
        self.hist_cmd[ptr] = cmd

        # EFeru main loop: rate limiter, then low-pass, on the delayed command.
        due = np.nonzero(self.t >= self.eferu_next)[0]
        if due.size:
            c_in = self.hist_cmd[(ptr - p["cmd_delay"][due]) % H, due]
            lim = p["rate"][due]
            self.u_rl[due] += np.clip(c_in - self.u_rl[due], -lim, lim)
            self.u_f[due] += p["cmd_lowpass"][due] * (self.u_rl[due] - self.u_f[due])
            self.eferu_next[due] += EFERU_LOOP_S

        # Hub motors. The rotor speed the motor sees is relative to its stator,
        # which is bolted to the body.
        u = np.clip(self.u_f / EFERU_FULL_SCALE, -1.0, 1.0)
        w = self.phid - self.thd
        vb = np.maximum(p["battery_ocv"] - p["battery_resistance"] * self.i_dc,
                        0.5 * p["battery_ocv"])
        self.v_batt = vb
        k, R, imax = p["k_m"], p["r_m"], p["imax_m"]
        emf = k * w
        e = u * p["spd_full"] - w
        self.spd_int = self.spd_int + e * dt
        wind = imax.min(axis=0) / np.maximum(p["spd_ki"], 1e-9)
        self.spd_int = np.clip(self.spd_int, -wind, wind)
        mode = p["mode"]
        i_cmd = np.where(mode == VOLTAGE, (u * vb - emf) / R,
                         np.where(mode == SPEED, p["speed_kp"] * e + p["spd_ki"] * self.spd_int,
                                  u * imax))
        i = np.clip(np.clip(i_cmd, -imax, imax), (-vb - emf) / R, (vb - emf) / R)
        tau_motor = (k * i).sum(axis=0)
        self.i_dc = ((R * i + emf) * i).sum(axis=0) / vb
        self.batt_lp += dt / BATT_FILTER_S * (vb - self.batt_lp)

        tau = (tau_motor - p["viscous_friction"] * w
               - p["coulomb_friction"] * np.tanh(w / COULOMB_EPS))

        # Rigid-body dynamics with the wheel-floor friction solved at velocity level.
        M, Iw, r = p["body_mass"], p["wheel_inertia"], p["wheel_radius"]
        L, mt, Jb, slope = p["L"], p["mt"], p["Jb"], p["slope"]
        psi = self.th + p["phi0"]
        s, c = np.sin(psi), np.cos(psi)
        sa, ca = np.sin(slope), np.cos(slope)
        MLc = M * L * c
        det = mt * Jb - MLc * MLc
        N = np.maximum(mt * G * ca - M * L * (s * self.psidd + c * self.thd ** 2), 0.0)
        push = np.where(self.t < self.push_until, self.push_F, 0.0)

        b_x = push + M * L * s * self.thd ** 2 - mt * G * sa
        b_psi = -tau + push * self.push_h * np.cos(self.th) + M * G * L * np.sin(psi - slope)
        b_phi = tau - p["rolling_resistance"] * N * r * np.tanh(self.phid / ROLL_EPS)

        k_c = Jb / det + r * r / Iw
        F_roll = (r * b_phi / Iw - (Jb * b_x - MLc * b_psi) / det) / k_c
        v_slip = self.xd - r * self.phid
        mu = np.where(np.abs(v_slip) < SLIP_EPS, p["mu_s"], p["mu_kinetic"])
        F = np.clip(F_roll - v_slip / (k_c * dt), -mu * N, mu * N)

        xdd = (Jb * (b_x + F) - MLc * b_psi) / det
        psidd = (mt * b_psi - MLc * (b_x + F)) / det
        phidd = (b_phi - F * r) / Iw
        self.xd += xdd * dt
        self.x += self.xd * dt
        self.thd += psidd * dt
        self.th += self.thd * dt
        self.phid += phidd * dt
        self.phi += self.phid * dt
        self.psidd = psidd

        # What the sensors would see now; delivered later through the histories.
        self.imu_err += dt / p["imu_fusion_s"] * (p["imu_accel_tilt"] * xdd / G - self.imu_err)
        h = p["imu_height"]
        ct, st = np.cos(self.th), np.sin(self.th)
        ax = xdd + h * (ct * psidd - st * self.thd ** 2)
        ay = -h * (st * psidd + ct * self.thd ** 2)
        self.hist_pitch[ptr] = self.th - slope + self.imu_err
        self.hist_accel[ptr] = (ax + G * sa) * ct - (ay + G * ca) * st
        self.rpm_lag += dt / p["fb_lag"] * ((self.phid - self.thd) * RPM_PER_RAD_S - self.rpm_lag)
        self.hist_rpm[ptr] = self.rpm_lag

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
            m = due.size
            self.push_F[due] = (rng.choice([-1.0, 1.0], m) * rng.uniform(0, 1, m)
                                * p["push_force_max"][due])
            self.push_until[due] = self.t[due] + rng.uniform(0.05, 0.4, m)
            self.push_h[due] = rng.uniform(0.3, 1.2, m) * p["com_height"][due]
            self.push_next[due] = self.t[due] + self._exp(p["push_rate"][due])

        # Sudden changes: new slope, floor patch, or payload shift. Body pitch
        # from vertical stays continuous across a slope change.
        due = np.nonzero(self.t >= self.change_next)[0]
        if due.size:
            old_slope = p["slope"][due].copy()
            new = sample(self.rand, rng, due.size, self.scale)
            for key in CHANGEABLE:
                p[key][due] = new[key]
            self._derive(due)
            self.th[due] += p["slope"][due] - old_slope
            self.change_next[due] = self.t[due] + self._exp(p["change_rate"][due])

        self.t += dt
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
        return {
            "pitch": self.th - self.p["slope"], "pitch_rate": self.thd, "speed": self.xd,
            "position": self.x, "wheel_angle": self.phi, "slip": self.xd - self.p["wheel_radius"] * self.phid,
            "slope": self.p["slope"], "push": np.where(self.t < self.push_until, self.push_F, 0.0),
            "command_at_motor": self.u_f, "battery_v": self.v_batt, "battery_a": self.i_dc,
        }
