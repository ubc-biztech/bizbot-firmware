"""Vectorised RL environment: N robots stepped together with numpy.

The policy plays the ESP32's part. Every `policy_dt` it gets what the ESP32
would have (IMU pitch and acceleration, hoverboard wheel rpm and battery
voltage, the operator's CMD_VEL linear command) and returns a normalised speed
command in [-1, 1], sent to the hoverboard as command * MAX_HOVERBOARD_COMMAND.
With lqr_base, the ESP32 also runs the LQR every control tick, tracking
CMD_VEL, and the policy's command times residual_scale is added to the LQR's.
Robots that fall or run out of time are reset automatically.
"""

from dataclasses import dataclass

import numpy as np

from .controllers import CONTROL_DT, lqr
from .params import Randomization
from .plant import G, RPM_PER_RAD_S, Plant


@dataclass
class EnvConfig:
    n_envs: int = 1024
    physics_dt: float = 0.00025
    policy_dt: float = 0.01
    episode_s: float = 20.0
    history: int = 8                 # policy steps of sensor history in each observation
    max_cmd: int = 500               # firmware MAX_HOVERBOARD_COMMAND
    max_wheel_rpm: float = 60.0      # firmware MAX_WHEEL_RPM, reached at CMD_VEL linear = 1
    max_tilt_deg: float = 35.0       # firmware MAX_TILT_DEG
    action_rate_limit: float = 0.25  # largest command change per policy step
    lqr_base: bool = False           # policy corrects the LQR instead of commanding alone
    residual_scale: float = 0.2      # largest correction, as a share of full command
    integral_leak_s: float = 20.0    # wheel-speed error integral forgets over this time
    integral_limit: float = 2.0      # wheel revolutions
    vref_change_rate: float = 0.3    # new CMD_VEL per second
    vref_zero_prob: float = 0.5      # share of CMD_VEL that are "stand still"
    init_pitch_deg: float = 5.0
    init_rate_dps: float = 20.0
    init_speed: float = 0.3          # m/s
    # Reward weights.
    w_alive: float = 1.0
    w_vel: float = 1.0
    w_rate: float = 0.05
    w_cmd: float = 0.05
    w_dcmd: float = 5.0
    w_hold: float = 1.0              # per wheel revolution^2 behind or ahead of CMD_VEL
    fall_penalty: float = 50.0


class BizBotEnv:
    FRAME = 4  # per history step: pitch, acceleration, wheel rpm, last command

    def __init__(self, cfg=EnvConfig(), rand=Randomization(), seed=None, scale=1.0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        n = cfg.n_envs
        self.substeps = int(round(cfg.policy_dt / cfg.physics_dt))
        self.plant = Plant(n, rand, cfg.physics_dt, self.rng, scale)
        self.obs_size = self.FRAME * cfg.history + 3
        self.hist = np.zeros((n, cfg.history, self.FRAME), dtype=np.float32)
        self.base = lqr(n) if cfg.lqr_base else None
        self.ticks = int(round(cfg.policy_dt / CONTROL_DT))
        self.u = np.zeros(n)
        self.sent = np.zeros(n)          # with lqr_base: LQR plus correction, last tick
        self.linear = np.zeros(n)
        self.vref_next = np.zeros(n)
        self.integral = np.zeros(n)
        self.ep_return = np.zeros(n)
        self.ep_len = np.zeros(n, dtype=np.int64)
        self._reset(np.arange(n))

    @property
    def scale(self):
        return self.plant.scale

    @scale.setter
    def scale(self, value):
        """Randomization width for robots reset from now on (0 nominal, 1 full)."""
        self.plant.scale = float(value)

    def reset(self):
        self._reset(np.arange(self.cfg.n_envs))
        return self._obs()

    def step(self, action):
        c, pl = self.cfg, self.plant
        a = np.clip(np.asarray(action, dtype=float).reshape(-1), -1.0, 1.0)
        u_prev = self.u
        self.u = u_prev + np.clip(a - u_prev, -c.action_rate_limit, c.action_rate_limit)
        if self.base is None:
            pl.advance(np.trunc(self.u * c.max_cmd), self.substeps)  # normalizedToHoverboard
        else:
            rpm_ref = self.linear * c.max_wheel_rpm
            for _ in range(self.ticks):
                lqr_u = self.base(pl.measurements(), rpm_ref=rpm_ref, sent=self.sent)
                self.sent = np.clip(lqr_u + c.residual_scale * self.u, -1.0, 1.0)
                pl.advance(np.trunc(self.sent * c.max_cmd), self.substeps // self.ticks)

        due = np.nonzero(pl.t >= self.vref_next)[0]
        if due.size:
            self.linear[due] = self._new_linear(due.size)
            self.vref_next[due] = pl.t[due] + self.rng.exponential(1.0, due.size) / c.vref_change_rate

        # Leaky integral of wheel-speed error, kept by the firmware: the policy's
        # only memory longer than its history, which lets it find its own trim.
        rpm = self._wheel_rpm()
        err = (self.linear * c.max_wheel_rpm - rpm) / 60.0
        self.integral = np.clip(self.integral * np.exp(-c.policy_dt / c.integral_leak_s)
                                + err * c.policy_dt, -c.integral_limit, c.integral_limit)

        self.hist[:, :-1] = self.hist[:, 1:]
        self.hist[:, -1] = self._frame(slice(None))

        truth = pl.truth()
        v_max = c.max_wheel_rpm / RPM_PER_RAD_S * pl.p["wheel_radius"]
        vel_err = (truth["speed"] - self.linear * v_max) / v_max
        terminated = np.abs(truth["pitch"]) > np.radians(c.max_tilt_deg)
        truncated = ~terminated & (pl.t >= c.episode_s - 1e-9)
        penalty = (c.w_vel * vel_err ** 2 + c.w_rate * truth["pitch_rate"] ** 2
                   + c.w_cmd * self.u ** 2 + c.w_dcmd * (self.u - u_prev) ** 2
                   + c.w_hold * self.integral ** 2)
        # Capped so that staying up never scores below falling.
        reward = c.w_alive - np.minimum(penalty, c.w_alive)
        reward = np.where(terminated, -c.fall_penalty, reward)
        self.ep_return += reward
        self.ep_len += 1

        obs = self._obs()
        info = {}
        done = np.nonzero(terminated | truncated)[0]
        if done.size:
            info = {"done": done, "terminal_obs": obs[done].copy(), "fell": terminated[done],
                    "episode_return": self.ep_return[done].copy(),
                    "episode_length": self.ep_len[done].copy()}
            self._reset(done)
            obs[done] = self._obs()[done]
        return obs, reward.astype(np.float32), terminated, truncated, info

    # --- internals -------------------------------------------------------------

    def _reset(self, idx):
        c, rng, m = self.cfg, self.rng, len(idx)
        self.plant.reset(idx,
                         pitch=np.radians(rng.uniform(-1, 1, m) * c.init_pitch_deg),
                         rate=np.radians(rng.uniform(-1, 1, m) * c.init_rate_dps),
                         speed=rng.uniform(-1, 1, m) * c.init_speed)
        self.u[idx] = 0.0
        self.sent[idx] = 0.0
        if self.base is not None:
            self.base.reset(idx)
        self.integral[idx] = 0.0
        self.ep_return[idx] = 0.0
        self.ep_len[idx] = 0
        self.linear[idx] = self._new_linear(m)
        self.vref_next[idx] = rng.exponential(1.0, m) / c.vref_change_rate
        self.hist[idx] = self._frame(idx)[:, None, :]

    def _new_linear(self, m):
        still = self.rng.random(m) < self.cfg.vref_zero_prob
        return np.where(still, 0.0, self.rng.uniform(-1, 1, m))

    def _wheel_rpm(self, idx=slice(None)):
        # Firmware: wheelVelocity is the mean of left and (sign-corrected) right.
        return 0.5 * (self.plant.fb_rpm[0, idx] + self.plant.fb_rpm[1, idx])

    def _frame(self, idx):
        pl, c = self.plant, self.cfg
        cmd = self.u if self.base is None else self.sent
        return np.stack([pl.imu_pitch[idx] / 10.0, pl.imu_accel[idx] / G,
                         self._wheel_rpm(idx) / c.max_wheel_rpm, cmd[idx]], axis=1)

    def _obs(self):
        pl, c = self.plant, self.cfg
        extra = np.stack([self.linear, (pl.batt_meas - 37.0) / 5.0,
                          self.integral / c.integral_limit], axis=1)
        return np.concatenate([self.hist.reshape(c.n_envs, -1), extra], axis=1).astype(np.float32)
