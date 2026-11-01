"""Ranges for everything the MuJoCo sim cannot know about the real robot.

Each robot draws its values at every reset. `scale` in [0, 1] shrinks every
range toward its nominal value: 0 is the nominal robot only, 1 the full range.
None of the nominal values are measured. They only set the middle of a range.

MuJoCo handles the rigid bodies, wheel-floor contact (friction, slip, tyre and
floor compliance) and drivetrain stiction. The hoverboard firmware, motors and
IMU are not physics an engine can know, so they are modelled here as broadly
as possible, including effects nobody has reported but that real robots have:
a body that flexes, an IMU fusion filter, a command deadband.
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
class P:
    """Bernoulli with probability p at full range, 0 at scale 0 (unless pinned)."""

    p: float
    pinned: bool = False

    def sample(self, rng, n, scale):
        return rng.random(n) < (self.p if self.pinned else self.p * scale)


@dataclass(frozen=True)
class Randomization:
    # --- body above the axle ---
    body_mass: U = U(10.0, 45.0, 20.0)              # kg, everything above the axle
    com_height: U = U(0.15, 0.9, 0.45)              # m above the axle, along the body
    com_offset: U = U(-0.08, 0.08, 0.0)             # m forward of the body axis
    body_gyration: U = U(0.08, 0.4, 0.2)            # m, pitch radius of gyration about the COM
    base_share: U = U(0.05, 0.4, 0.2)               # share of body_mass in the frame at the axle
    # The upper body is joined to the frame by a pitch hinge at flex_height *
    # com_height. Stiff robots lock it; the rest have a spring there.
    flex: P = P(0.5)
    flex_hz: U = U(3.0, 20.0, 8.0, log=True)        # natural frequency of the upper body on its spring
    flex_zeta: U = U(0.02, 0.3, 0.08, log=True)     # damping ratio
    flex_height: U = U(0.0, 0.6, 0.2)               # hinge height as a share of com_height
    track: U = U(0.35, 0.6, 0.48)                   # m between wheel centres

    # --- wheels ---
    wheel_mass: U = U(5.0, 9.0, 7.0)                # kg, both hub motors
    wheel_inertia: U = U(0.01, 0.04, 0.02)          # kg m^2, both rotors
    wheel_radius: U = U(0.080, 0.085, 0.0825)       # m

    # --- floor and tyres ---
    mu: U = U(0.2, 1.0, 0.7)                        # wheel-floor friction
    contact_tc: U = U(0.004, 0.03, 0.01, log=True)  # s, MuJoCo solref time constant (tyre and floor give)
    contact_damp: U = U(0.3, 1.5, 1.0)              # MuJoCo solref damping ratio
    rolling_resistance: U = U(0.0, 0.03, 0.01)      # coefficient
    slope_deg: U = U(-6.0, 6.0, 0.0)                # + = uphill ahead

    # --- drivetrain friction (rotor relative to the body) ---
    coulomb_friction: U = U(0.0, 1.5, 0.3)          # N m, both motors; true stiction in MuJoCo
    viscous_friction: U = U(0.0, 0.05, 0.01)        # N m s/rad, both motors

    # --- hub motors (per motor; left and right differ by up to motor_mismatch) ---
    back_emf: U = U(0.6, 1.2, 0.9)                  # V s/rad, equal to N m/A
    phase_resistance: U = U(0.15, 0.6, 0.3, log=True)  # ohm
    current_limit: U = U(8.0, 20.0, 15.0)           # A
    motor_mismatch: U = U(0.0, 0.25, 0.1)           # relative spread of the three above

    # --- battery ---
    battery_ocv: U = U(32.0, 42.0, 38.0)            # V
    battery_resistance: U = U(0.05, 0.4, 0.15)      # ohm

    # --- hoverboard firmware, speed mode (EFeru); all unknown ---
    cmd_rate_limit: U = U(5.0, 300.0, 30.0, log=True)  # command units per 5 ms loop
    cmd_rate_limit_off: P = P(0.2)
    cmd_lowpass: U = U(0.03, 1.0, 0.1, log=True)    # per 5 ms loop; 1 = no filter
    cmd_deadband: U = U(0.0, 15.0, 0.0)             # command units read as 0
    speed_full_scale_rpm: U = U(400.0, 1000.0, 1000.0)  # rpm at command 1000
    speed_kp: U = U(0.5, 6.0, 2.0, log=True)        # A per rad/s
    speed_ki_ratio: U = U(0.0, 10.0, 3.0)           # ki / kp, 1/s

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
    imu_latency: U = U(0.0, 0.08, 0.01)             # s, pure delay, sample to ESP32
    imu_filter: U = U(0.0, 0.03, 0.005)             # s, first-order lag inside the fusion
    imu_jitter: U = U(0.0, 0.006, 0.001)            # s, extra per-packet age, uniform up to this
    imu_noise_deg: U = U(0.002, 0.5, 0.03, log=True)
    imu_mount_deg: U = U(-3.0, 3.0, 0.0)            # mounting and trim error
    imu_accel_tilt: U = U(-0.3, 0.3, 0.0)           # fusion pitch error per g of acceleration
    imu_fusion_s: U = U(0.2, 3.0, 1.0, log=True)    # s for that error to build up
    imu_height: U = U(0.05, 0.6, 0.25)              # m up the body; above the flex hinge it rides the upper body
    accel_noise: U = U(0.02, 0.5, 0.1, log=True)    # m/s^2
    imu_drop_prob: U = U(0.0, 0.05, 0.005)

    # --- disturbances ---
    push_rate: U = U(0.0, 0.5, 0.1)                 # pushes per second
    push_force_max: U = U(20.0, 200.0, 80.0)        # N
    change_rate: U = U(0.0, 0.2, 0.05)              # sudden slope, floor or payload changes per second

    # --- seen on the real robot (2026-10-23 logs); all off at nominal ---
    cmd_deadzone: P = P(0.0)                        # the deadband is also taken off larger commands (dead zone), not a hard cut
    hall_speed: P = P(0.0)                          # the hoverboard speed estimate only changes at hall edges (stale when slow)
    hall_edges: U = U(45.0, 135.0, 90.0)            # hall edges per wheel turn
    hall_timeout: U = U(0.05, 0.3, 0.15)            # s without an edge before the estimate reads 0
    hand_rate: U = U(0.0, 0.0, 0.0)                 # times per second someone holds, tilts or lifts the robot
    hand_lift_share: U = U(0.0, 1.0, 0.5)           # share of those that lift the wheels off the floor
    floor_spring_on: P = P(0.0)                     # a carpet-like pull back toward a resting spot that follows the robot
    floor_spring: U = U(0.0, 0.0, 0.0)              # N/m of that pull
    floor_spring_tau: U = U(10.0, 10.0, 10.0)       # s for the resting spot to catch up with the robot
    voltage_mode: P = P(0.0)                        # EFeru voltage mode: command sets motor voltage (no speed loop)
    cmd_limit: U = U(1000.0, 1000.0, 1000.0)       # largest command magnitude the hoverboard acts on (real: about 380)
    hall_jitter: U = U(0.0, 0.0, 0.0)               # relative error of each hall-edge speed estimate (uneven edge spacing)


def sample(rand, rng, n, scale):
    """Draw n robots: {field name: array}."""
    return {f.name: getattr(rand, f.name).sample(rng, n, scale) for f in fields(rand)}


def fixed(rand=None, **values):
    """A Randomization with the named fields pinned to one value each."""
    from dataclasses import replace
    rand = rand or Randomization()
    out = {}
    for k, v in values.items():
        cur = getattr(rand, k)
        out[k] = P(float(v), pinned=True) if isinstance(cur, P) else U(v, v, v)
    return replace(rand, **out)
