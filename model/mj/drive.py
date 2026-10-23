from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class DriveLimits:
    speed: float = 0.10
    acceleration: float = 0.06
    braking: float = 0.08
    recovery_pitch_deg: float = 12.0
    recovery_pitch_rate_dps: float = 75.0
    recovery_command: float = 0.90


class DriveController:
    def __init__(self, dt, history, integral_leak_s, integral_limit, limits=DriveLimits()):
        self.dt = dt
        self.history = history
        self.integral_leak_s = integral_leak_s
        self.integral_limit = integral_limit
        self.limits = limits
        self.reset()

    def set_speed_limit(self, speed):
        if not np.isfinite(speed) or not 0 <= speed <= 10.0:
            raise ValueError('Speed request limit must be between 0 and 10.00 m/s')
        self.limits = replace(self.limits, speed=float(speed))

    def set_acceleration_limit(self, acceleration):
        if not np.isfinite(acceleration) or not .01 <= acceleration <= 50.0:
            raise ValueError('Acceleration limit must be between 0.01 and 50.00 m/s²')
        self.limits = replace(self.limits, acceleration=float(acceleration))

    def reset(self, require_release=False):
        self.reference = 0.0
        self.acceleration = 0.0
        self.reference_history = np.zeros(self.history)
        self.integral = 0.0
        self.engaged = False
        self.require_release = require_release
        self.measured_speed = 0.0
        self.status = 'Release WASD to rearm' if require_release else 'Balancing'

    def adapt(self, observation, wheel_radius, direction, turning=False):
        frame = observation[0, :-2].reshape(self.history, 4)
        self.measured_speed = float(frame[-1, 2] * 2 * np.pi * wheel_radius)
        pitch = float(frame[-1, 0] * 10)
        span = min(5, self.history - 1)
        pitch_rate = float((frame[-1, 0] - frame[-1-span, 0]) * 10 / (span * self.dt)) if span else 0.0
        command = float(frame[-1, 3])
        limits = self.limits
        unsafe = (abs(pitch) > limits.recovery_pitch_deg
                  or abs(pitch_rate) > limits.recovery_pitch_rate_dps
                  or abs(command) > limits.recovery_command)
        if unsafe and (direction or self.engaged or turning):
            self.require_release = True
        if self.require_release:
            if direction == 0 and not turning and not unsafe:
                self.require_release = False
            else:
                direction = 0
        target = direction * limits.speed if not self.require_release else 0.0
        if target * self.reference < 0 and abs(self.measured_speed) > .04:
            target = 0.0
        if direction and not self.require_release and not self.engaged:
            self.engaged = True
            self.integral = float(observation[0, -1] * self.integral_limit)
        if not self.engaged:
            self.status = 'Release WASD to rearm' if self.require_release else 'Balancing'
            return observation
        error = target - self.reference
        braking = abs(target) < abs(self.reference) or target * self.reference < 0
        limit = max(limits.braking, limits.acceleration) if braking else limits.acceleration
        change = float(np.clip(error, -limit * self.dt, limit * self.dt))
        if direction * self.measured_speed >= limits.speed and direction * change > 0:
            change = 0.0
        self.acceleration = change / self.dt
        self.reference = target if change == error else self.reference + change
        adapted = observation.copy()
        adapted[0, :-2].reshape(self.history, 4)[:, 2] -= self.reference_history
        adapted[0, -1] = self.integral / self.integral_limit
        self.status = ('Release WASD to rearm' if self.require_release else
                       'Driving' if direction else
                       'Stopping' if abs(self.measured_speed) > .04 or abs(self.reference) > .001 else 'Balancing')
        return adapted

    def record(self, observation, wheel_radius):
        if not self.engaged:
            return
        reference_rps = self.reference / (2 * np.pi * wheel_radius)
        self.reference_history[:-1] = self.reference_history[1:]
        self.reference_history[-1] = reference_rps
        actual_rps = float(observation[0, :-2].reshape(self.history, 4)[-1, 2])
        self.integral = float(np.clip(self.integral * np.exp(-self.dt / self.integral_leak_s)
                                     - (actual_rps - reference_rps) * self.dt,
                                     -self.integral_limit, self.integral_limit))

    def state(self):
        return {'reference': self.reference, 'speed': self.measured_speed,
                'limit': self.limits.speed, 'acceleration_limit': self.limits.acceleration,
                'status': self.status}
