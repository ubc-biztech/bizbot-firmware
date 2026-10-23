import mujoco
import numpy as np

from bizbot_mj.params import sample
from bizbot_mj.plant import (Plant, G, MG, DEG, RPM_PER_RAD_S, EFERU_LOOP_S,
                             EFERU_FULL_SCALE, BATT_FILTER_S, ROLL_EPS, NV, CHANGEABLE)


class TurningPlant(Plant):
    @classmethod
    def from_plant(cls, plant, command_limit):
        result = cls.__new__(cls)
        result.__dict__ = plant.__dict__.copy()
        result.command_limit = command_limit
        for name in ('turn_command', 'turn_rl', 'turn_f', 'turn_rpm_lag'):
            setattr(result, name, np.zeros(plant.n))
        result.hist_turn_cmd = np.zeros((plant.H, plant.n))
        result.hist_turn_rpm = np.zeros((plant.H, plant.n))
        return result

    def reset(self, idx, pitch, rate, speed):
        super().reset(idx, pitch, rate, speed)
        for name in ('turn_command', 'turn_rl', 'turn_f', 'turn_rpm_lag'):
            getattr(self, name)[idx] = 0.0
        self.hist_turn_cmd[:, idx] = 0.0
        self.hist_turn_rpm[:, idx] = 0.0

    def heading(self):
        w, x, y, z = self.qpos[:, 3:7].T
        heading = np.stack([1 - 2*(y*y + z*z), 2*(x*y + w*z)], axis=1)
        return heading / np.maximum(np.linalg.norm(heading, axis=1, keepdims=True), 1e-9)

    def wheel_axis(self):
        w, x, y, z = self.qpos[:, 3:7].T
        return np.stack([2*(x*y - w*z), 1 - 2*(x*x + z*z), 2*(y*z + w*x)], axis=1)

    def base_pitch(self):
        w, x, y, z = self.qpos[:, 3:7].T
        up_xy = np.stack([2*(x*z + w*y), 2*(y*z - w*x)], axis=1)
        return np.arctan2((up_xy * self.heading()).sum(axis=1), 1 - 2*(x*x + y*y))

    def _substep(self, cmd):
        p, dt, H, rng, ptr, m, d = self.p, self.dt, self.H, self.rng, self.ptr, self.m, self.d
        self.hist_cmd[ptr] = cmd
        self.hist_turn_cmd[ptr] = self.turn_command

        # EFeru main loop: deadband, rate limiter, low-pass on the delayed command.
        due = np.nonzero(self.t >= self.eferu_next)[0]
        if due.size:
            c_in = self.hist_cmd[(ptr - p["cmd_delay"][due]) % H, due]
            c_in = np.where(np.abs(c_in) <= p["cmd_deadband"][due], 0.0, c_in)
            lim = p["rate"][due]
            self.u_rl[due] += np.clip(c_in - self.u_rl[due], -lim, lim)
            self.u_f[due] += p["cmd_lowpass"][due] * (self.u_rl[due] - self.u_f[due])
            turn_in = self.hist_turn_cmd[(ptr - p["cmd_delay"][due]) % H, due]
            self.turn_rl[due] += np.clip(turn_in - self.turn_rl[due], -lim, lim)
            self.turn_f[due] += p["cmd_lowpass"][due] * (self.turn_rl[due] - self.turn_f[due])
            self.eferu_next[due] += EFERU_LOOP_S

        # Hub motors in speed mode: i = kp (target - w) + ki * integral, the
        # kp * w part being the actuator's implicit damping.
        qv = self.qvel
        w = np.stack([qv[:, 7], qv[:, 8]])                       # rotor speed relative to the frame
        room = np.maximum(0.0, self.command_limit - np.abs(self.u_f))
        differential = np.clip(self.turn_f, -room, room)
        wheel_commands = np.stack([self.u_f - differential, self.u_f + differential])
        target = np.clip(wheel_commands / EFERU_FULL_SCALE, -1.0, 1.0) * p["spd_full"]
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
            q = self.qpos[:, 3:7]
            qw, qx, qy, qz = q.T
            xf[self.b_up, 4] = F * arm * (1 - 2*(qx*qx + qy*qy))
            xf[self.b_up, 5] = -F * arm * 2*(qy*qz - qw*qx)
            self.push_F = np.where(pushing, self.push_F, 0.0)
        w_abs_l = w[0] + qv[:, 4]
        w_abs_r = w[1] + qv[:, 4]
        rr = p["rolling_resistance"] * p["w_load"] * p["wheel_radius"]
        axis = self.wheel_axis()
        d.xfrc_applied[self.b_wl, 3:6] = (-rr * np.tanh(w_abs_l / ROLL_EPS))[:, None] * axis
        d.xfrc_applied[self.b_wr, 3:6] = (-rr * np.tanh(w_abs_r / ROLL_EPS))[:, None] * axis

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
        heading = self.heading()
        acc_w = (d.qacc.reshape(self.n, NV)[:, :2] * heading).sum(axis=1)
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
        turn_rpm = .5 * (qv[:, 8] - qv[:, 7]) * RPM_PER_RAD_S
        self.turn_rpm_lag += dt / p["fb_lag"] * (turn_rpm - self.turn_rpm_lag)
        self.hist_turn_rpm[ptr] = self.turn_rpm_lag
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
            rows = (ptr - p["fb_delay"][due]) % H
            common_rpm = self.hist_rpm[rows, due]
            differential_rpm = self.hist_turn_rpm[rows, due]
            rpm = np.stack([common_rpm - differential_rpm, common_rpm + differential_rpm]) * p["fb_gain"][:, due]
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

