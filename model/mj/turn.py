import numpy as np


class TurnController:
    def __init__(self, dt):
        self.dt = dt
        self.rate_limit = .25
        self.acceleration_limit = .5
        self.command_fraction = .15
        self.reset()

    def set_rate_limit(self, degrees_per_second):
        if not np.isfinite(degrees_per_second) or not 0 <= degrees_per_second <= 720:
            raise ValueError('Turn-rate limit must be between 0 and 720 degrees/s')
        self.rate_limit = float(np.radians(degrees_per_second))

    def set_acceleration_limit(self, degrees_per_second_squared):
        if not np.isfinite(degrees_per_second_squared) or not 1 <= degrees_per_second_squared <= 3600:
            raise ValueError('Turn acceleration must be between 1 and 3600 degrees/s²')
        self.acceleration_limit = float(np.radians(degrees_per_second_squared))

    def set_command_limit(self, percent):
        if not np.isfinite(percent) or not 0 <= percent <= 100:
            raise ValueError('Steering command cap must be between 0 and 100 percent')
        self.command_fraction = float(percent) / 100

    def reset(self):
        self.reference = 0.0
        self.integral = 0.0
        self.measured = 0.0
        self.engaged = False

    def update(self, request, drive, rpm_left, rpm_right, radius, track, full_rpm, command_limit):
        self.measured = float((rpm_right - rpm_left) * (2*np.pi/60) * radius / track)
        if request:
            self.engaged = True
        if not self.engaged:
            return 0.0
        target = request * self.rate_limit if not drive.require_release else 0.0
        step = self.acceleration_limit * self.dt
        self.reference += float(np.clip(target - self.reference, -step, step))
        if drive.require_release:
            self.reset()
            return 0.0
        error = self.reference - self.measured
        self.integral = float(np.clip(self.integral + error*self.dt, -.4, .4))
        feedforward = self.reference * track / (2*radius) * 60 / (2*np.pi) / full_rpm * 1000
        cap = self.command_fraction * command_limit
        command = float(np.clip(feedforward + 25*error + 15*self.integral, -cap, cap))
        if not request and abs(self.reference) < 1e-9 and abs(self.measured) < .02:
            self.integral = 0.0
            self.engaged = False
            return 0.0
        return command

    def state(self):
        return {'reference': self.reference, 'rate': self.measured,
                'limit_deg': float(np.degrees(self.rate_limit)),
                'acceleration_deg': float(np.degrees(self.acceleration_limit)),
                'command_percent': self.command_fraction * 100}
