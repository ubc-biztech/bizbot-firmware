"""Ranges for everything the sim cannot know about the real robot.

Each robot draws its values from these ranges at every reset. The ranges are
deliberately wider than what we expect, on both the better and the worse side.
`scale` in [0, 1] shrinks every range toward its nominal value: 0 is the
nominal robot only, 1 is the full range.

Nominal values are placeholders until measured.
"""

from dataclasses import dataclass, fields

import numpy as np


@dataclass(frozen=True)
class U:
    """Uniform on [lo, hi], or log-uniform when log=True."""

    lo: float
    hi: float
    nominal: float
    log: bool = False

    def sample(self, rng, n, scale):
        if self.log:
            c = np.log(self.nominal)
            lo = c + scale * (np.log(self.lo) - c)
            hi = c + scale * (np.log(self.hi) - c)
            return np.exp(rng.uniform(lo, hi, n))
        lo = self.nominal + scale * (self.lo - self.nominal)
        hi = self.nominal + scale * (self.hi - self.nominal)
        return rng.uniform(lo, hi, n)


@dataclass(frozen=True)
class Randomization:
    # --- body and payload (weight distribution) ---
    body_mass: U = U(10.0, 45.0, 20.0)              # kg, everything above the axle
    com_height: U = U(0.15, 0.9, 0.45)              # m up the body axis
    com_offset: U = U(-0.08, 0.08, 0.0)             # m forward of the body axis
    body_gyration: U = U(0.08, 0.4, 0.2)            # m, radius of gyration about the COM
    wheel_mass: U = U(5.0, 9.0, 7.0)                # kg, both hub motors
    wheel_inertia: U = U(0.01, 0.04, 0.02)          # kg m^2, both rotors
    wheel_radius: U = U(0.080, 0.085, 0.0825)       # m

    # --- floor ---
    mu_kinetic: U = U(0.2, 1.0, 0.7)
    mu_static_ratio: U = U(1.0, 1.4, 1.15)          # mu_static / mu_kinetic
    rolling_resistance: U = U(0.0, 0.03, 0.01)
    slope_deg: U = U(-6.0, 6.0, 0.0)                # + = uphill ahead

    # --- drivetrain friction (rotor relative to the body) ---
    coulomb_friction: U = U(0.0, 1.5, 0.3)          # N m, both motors
    viscous_friction: U = U(0.0, 0.05, 0.01)        # N m s/rad, both motors

    # --- hub motors (per motor; left and right differ by up to motor_mismatch) ---
    back_emf: U = U(0.6, 1.2, 0.9)                  # V s/rad, equal to N m/A
    phase_resistance: U = U(0.15, 0.6, 0.3, log=True)  # ohm
    current_limit: U = U(8.0, 20.0, 15.0)           # A
    motor_mismatch: U = U(0.0, 0.25, 0.1)           # relative spread of the three above

    # --- battery ---
    battery_ocv: U = U(32.0, 42.0, 38.0)            # V, empty to full 10S pack
    battery_resistance: U = U(0.05, 0.4, 0.15)      # ohm

    # --- hoverboard firmware (EFeru) ---
    # Control mode probabilities: voltage, speed, torque. At scale 0 the most
    # likely mode is used.
    mode_probs: tuple = (0.0, 1.0, 0.0)
    cmd_rate_limit: U = U(5.0, 300.0, 30.0, log=True)  # cmd units per 5 ms loop
    cmd_rate_limit_off_prob: float = 0.2
    cmd_lowpass: U = U(0.03, 1.0, 0.1, log=True)    # per 5 ms loop; 1 = no filter
    speed_full_scale_rpm: U = U(400.0, 1000.0, 1000.0)  # speed mode: rpm at cmd 1000
    speed_kp: U = U(0.5, 6.0, 2.0, log=True)        # speed mode: A per rad/s
    speed_ki_ratio: U = U(0.0, 10.0, 3.0)           # speed mode: ki / kp, 1/s

    # --- command and wheel feedback links ---
    cmd_latency: U = U(0.002, 0.04, 0.008)          # s, ESP32 command to hoverboard
    fb_period: U = U(0.008, 0.02, 0.01)             # s between feedback frames
    fb_latency: U = U(0.002, 0.04, 0.008)           # s
    fb_lag: U = U(0.002, 0.05, 0.01, log=True)      # s, hoverboard speed estimate filter
    fb_drop_prob: U = U(0.0, 0.1, 0.01)
    fb_min_rpm: U = U(0.0, 8.0, 2.0)                # slower than this reads as 0
    fb_gain_error: U = U(0.0, 0.05, 0.01)           # per motor, relative

    # --- IMU (BNO085 UART-RVC) ---
    imu_period: U = U(0.0095, 0.0105, 0.01)         # s
    imu_latency: U = U(0.0, 0.08, 0.01)             # s, sample to ESP32
    imu_jitter: U = U(0.0, 0.006, 0.001)            # s, extra per-packet sample age, uniform up to this
    imu_noise_deg: U = U(0.002, 0.5, 0.03, log=True)
    imu_mount_deg: U = U(-3.0, 3.0, 0.0)            # mounting and trim error
    imu_accel_tilt: U = U(-0.3, 0.3, 0.0)           # fusion pitch error per g of acceleration
    imu_fusion_s: U = U(0.2, 3.0, 1.0, log=True)    # s for that error to build up
    imu_height: U = U(0.05, 0.6, 0.25)              # m up the body axis
    accel_noise: U = U(0.02, 0.5, 0.1, log=True)    # m/s^2
    imu_drop_prob: U = U(0.0, 0.05, 0.005)

    # --- disturbances ---
    push_rate: U = U(0.0, 0.5, 0.1)                 # pushes per second
    push_force_max: U = U(20.0, 200.0, 80.0)        # N
    change_rate: U = U(0.0, 0.2, 0.05)              # sudden slope, floor or payload changes per second


def sample(rand, rng, n, scale):
    """Draw n robots. Returns {field name: array} plus 'mode' and 'rate_limit_off'."""
    out = {f.name: getattr(rand, f.name).sample(rng, n, scale)
           for f in fields(rand) if isinstance(getattr(rand, f.name), U)}
    probs = np.asarray(rand.mode_probs, dtype=float)
    if scale > 0:
        out["mode"] = rng.choice(len(probs), n, p=probs / probs.sum())
        out["rate_limit_off"] = rng.random(n) < rand.cmd_rate_limit_off_prob * scale
    else:
        out["mode"] = np.full(n, int(np.argmax(probs)))
        out["rate_limit_off"] = np.zeros(n, dtype=bool)
    return out
