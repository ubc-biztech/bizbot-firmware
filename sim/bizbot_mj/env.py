"""Vectorised balance task on the MuJoCo plant: n robots stepped together.

The policy plays the ESP32's part. Every `policy_dt` it gets what the ESP32
would have (IMU pitch and acceleration, mean wheel rpm, its own last command,
battery voltage, and a leaky integral of wheel rpm the firmware keeps) and
returns a normalised speed command in [-1, 1], sent to the hoverboard as
command * max_cmd. The task is to stay up, stand still and not wobble.

`priv` holds the true state and the robot's parameters, for a critic that is
only used in training (asymmetric actor-critic). The robot never has it.
"""

from dataclasses import dataclass

import numpy as np

from .params import Randomization
from .plant import G, RPM_PER_RAD_S, Plant


@dataclass
class EnvConfig:
    n_envs: int = 256
    physics_dt: float = 0.002
    policy_dt: float = 0.01
    episode_s: float = 20.0
    history: int = 16                # policy steps of sensor history in each observation
    max_cmd: int = 500               # firmware MAX_HOVERBOARD_COMMAND
    max_tilt_deg: float = 35.0       # firmware MAX_TILT_DEG
    action_rate_limit: float = 0.25  # largest command change per policy step
    integral_leak_s: float = 20.0    # wheel-rpm integral forgets over this time
    integral_limit: float = 2.0      # wheel revolutions
    init_pitch_deg: float = 5.0
    init_rate_dps: float = 20.0
    init_speed: float = 0.3          # m/s
    # Reward: penalty = w_vel (v / v_scale)^2 + w_rate pitch_rate^2 + w_dcmd dcmd^2 + w_cmd cmd^2
    reward_form: str = "exp"         # "capped": 1 - min(pen, 1); "exp": exp(-pen); "linear": 1 - pen;
                                     # "alive": 1 - shape_weight * (1 - exp(-pen)), so a step up always earns
                                     # at least 1 - shape_weight
    shape_weight: float = 0.5
    # "l1": 1 - l1_vel min(|v| / 1.5, 1) - l1_rate min(|pitch rate| / 2, 1) - l1_dcmd min(|dcmd| / 0.25, 1):
    # constant pull toward zero over each range, and a step up never earns less than 0.
    l1_vel: float = 0.6
    l1_rate: float = 0.2
    l1_dcmd: float = 0.2
    l1_v_scale: float = 1.5          # m/s at which the l1 speed penalty is full
    l1_rate_scale: float = 2.0       # rad/s at which the l1 pitch-rate penalty is full
    l1_hold: float = 0.0             # weight on the leaky wheel-rotation integral (distance crept)
    l1_hold_scale: float = 1.0       # wheel revolutions at which that penalty is full
    w_vel: float = 1.0
    v_scale: float = 0.3             # m/s
    w_rate: float = 0.05             # per (rad/s)^2
    w_dcmd: float = 5.0
    w_cmd: float = 0.0
    fall_penalty: float = 50.0


PRIV_PARAMS = ("body_mass", "com_height", "com_offset", "body_gyration", "flex", "flex_hz",
               "wheel_inertia", "mu", "slope_deg", "coulomb_friction", "back_emf", "current_limit",
               "speed_full_scale_rpm", "speed_kp", "speed_ki_ratio", "cmd_rate_limit", "cmd_lowpass",
               "cmd_deadband", "cmd_latency", "fb_latency", "fb_lag", "imu_latency", "imu_filter",
               "imu_noise_deg", "imu_mount_deg", "imu_accel_tilt", "imu_height")


class BalanceEnv:
    FRAME = 4  # per history step: pitch, acceleration, wheel rpm, last command

    def __init__(self, cfg=EnvConfig(), rand=Randomization(), seed=None, scale=1.0):
        self.cfg = cfg
        self.rand = rand
        self.rng = np.random.default_rng(seed)
        n = cfg.n_envs
        self.substeps = int(round(cfg.policy_dt / cfg.physics_dt))
        self.plant = Plant(n, rand, cfg.physics_dt, self.rng, scale)
        self.obs_size = self.FRAME * cfg.history + 2
        self.priv_size = 10 + len(PRIV_PARAMS)
        self.hist = np.zeros((n, cfg.history, self.FRAME), dtype=np.float32)
        self.u = np.zeros(n)
        self.integral = np.zeros(n)
        self.ep_return = np.zeros(n)
        self.ep_len = np.zeros(n, dtype=np.int64)
        self._lo = np.array([getattr(rand, k).lo if hasattr(getattr(rand, k), "lo") else 0.0 for k in PRIV_PARAMS])
        self._hi = np.array([getattr(rand, k).hi if hasattr(getattr(rand, k), "hi") else 1.0 for k in PRIV_PARAMS])
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
        pl.advance(np.trunc(self.u * c.max_cmd), self.substeps)  # normalizedToHoverboard

        rpm = self._wheel_rpm()
        self.integral = np.clip(self.integral * np.exp(-c.policy_dt / c.integral_leak_s)
                                - rpm / 60.0 * c.policy_dt, -c.integral_limit, c.integral_limit)
        self.hist[:, :-1] = self.hist[:, 1:]
        self.hist[:, -1] = self._frame(slice(None))

        truth = pl.truth()
        terminated = np.abs(truth["pitch"]) > np.radians(c.max_tilt_deg)
        terminated |= ~np.isfinite(truth["pitch"])
        truncated = ~terminated & (pl.t >= c.episode_s - 1e-9)
        pen = (c.w_vel * (truth["speed"] / c.v_scale) ** 2 + c.w_rate * truth["pitch_rate"] ** 2
               + c.w_dcmd * (self.u - u_prev) ** 2 + c.w_cmd * self.u ** 2)
        if c.reward_form == "capped":
            reward = 1.0 - np.minimum(pen, 1.0)
        elif c.reward_form == "exp":
            reward = np.exp(-pen)
        elif c.reward_form == "alive":
            reward = 1.0 - c.shape_weight * (1.0 - np.exp(-pen))
        elif c.reward_form == "l1":
            reward = 1.0 - (c.l1_vel * np.minimum(np.abs(truth["speed"]) / c.l1_v_scale, 1.0)
                            + c.l1_rate * np.minimum(np.abs(truth["pitch_rate"]) / c.l1_rate_scale, 1.0)
                            + c.l1_dcmd * np.minimum(np.abs(self.u - u_prev) / 0.25, 1.0)
                            + c.l1_hold * np.minimum(np.abs(self.integral) / c.l1_hold_scale, 1.0))
        else:
            reward = 1.0 - pen
        reward = np.where(terminated, -c.fall_penalty, reward)
        self.ep_return += reward
        self.ep_len += 1

        obs = self._obs()
        priv = self._priv(truth)
        info = {}
        done = np.nonzero(terminated | truncated)[0]
        if done.size:
            info = {"done": done, "terminal_obs": obs[done].copy(), "terminal_priv": priv[done].copy(),
                    "fell": terminated[done], "episode_return": self.ep_return[done].copy(),
                    "episode_length": self.ep_len[done].copy()}
            self._reset(done)
            obs[done] = self._obs()[done]
            priv[done] = self._priv(pl.truth())[done]
        self.last_priv = priv
        return obs, reward.astype(np.float32), terminated, truncated, info

    # --- internals -------------------------------------------------------------

    def _reset(self, idx):
        c, rng, m = self.cfg, self.rng, len(idx)
        self.plant.reset(idx,
                         pitch=np.radians(rng.uniform(-1, 1, m) * c.init_pitch_deg),
                         rate=np.radians(rng.uniform(-1, 1, m) * c.init_rate_dps),
                         speed=rng.uniform(-1, 1, m) * c.init_speed)
        self.u[idx] = 0.0
        self.integral[idx] = 0.0
        self.ep_return[idx] = 0.0
        self.ep_len[idx] = 0
        self.hist[idx] = self._frame(idx)[:, None, :]
        self.last_priv = self._priv(self.plant.truth())

    def _wheel_rpm(self, idx=slice(None)):
        # Firmware: wheelVelocity is the mean of left and (sign-corrected) right.
        return 0.5 * (self.plant.fb_rpm[0, idx] + self.plant.fb_rpm[1, idx])

    def _frame(self, idx):
        pl = self.plant
        return np.stack([pl.imu_pitch[idx] / 10.0, pl.imu_accel[idx] / G,
                         self._wheel_rpm(idx) / 60.0, self.u[idx]], axis=1)

    def _obs(self):
        pl, c = self.plant, self.cfg
        extra = np.stack([(pl.batt_meas - 37.0) / 5.0, self.integral / c.integral_limit], axis=1)
        return np.concatenate([self.hist.reshape(c.n_envs, -1), extra], axis=1).astype(np.float32)

    def _priv(self, truth):
        pl = self.plant
        state = np.stack([truth["pitch"] * 5.0, truth["pitch_rate"], truth["speed"] / 0.3,
                          truth["wheel_rate"] * pl.p["wheel_radius"] / 0.3, truth["flex"] * 20.0,
                          truth["flex_rate"], truth["push"] / 100.0,
                          truth["command_at_motor"] / 500.0, pl.u_rl / 500.0,
                          self.u], axis=1)
        params = np.stack([np.asarray(pl.p[k], dtype=float) for k in PRIV_PARAMS], axis=1)
        params = 2.0 * (params - self._lo) / np.maximum(self._hi - self._lo, 1e-9) - 1.0
        return np.concatenate([state, params], axis=1).astype(np.float32)
