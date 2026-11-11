import numpy as np

METHODS = ('push', 'creep', 'tilt', 'push+creep', 'push+tilt', 'creep+tilt', 'push+creep+tilt')


class PushDrive:
    """WASD driving on top of the policy, by one of METHODS (any mix of these) while W/S is held.

    push   both wheels -+command (inverted on purpose: pushing the wheels back tips the body
           forward and the policy drives forward under it)
    creep  creep integral set to +-creep_bias revolutions: the policy thinks it is behind where it
           started and drives forward to get back
    tilt   IMU pitch offset by -+tilt_bias degrees (like SET_TRIM): the policy thinks it leans back
           and leans forward
    With push or creep, creep is held (at zero without creep) while driving and after release until
    the wheels stop; tilt alone leaves creep alone.
    A/D in every method: right wheel +-turn_command, left wheel -+turn_command."""

    STOPPED_MPS = 0.02

    def __init__(self, command=80.0, turn_command=100.0, creep_bias=1.0, tilt_bias=1.0, method='push+creep+tilt'):
        self.command = command
        self.turn_command = turn_command
        self.creep_bias = creep_bias
        self.tilt_bias = tilt_bias
        self.method = method
        self.require_release = False
        self.reset()

    def set_command(self, command):
        if not np.isfinite(command) or not 0 <= command <= 200:
            raise ValueError('Push command must be between 0 and 200')
        self.command = float(command)

    def set_turn_command(self, command):
        if not np.isfinite(command) or not 0 <= command <= 200:
            raise ValueError('Turn command must be between 0 and 200')
        self.turn_command = float(command)

    def set_creep_bias(self, revolutions):
        if not np.isfinite(revolutions) or not 0 <= revolutions <= 2:
            raise ValueError('Creep bias must be between 0 and 2 revolutions')
        self.creep_bias = float(revolutions)

    def set_tilt_bias(self, degrees):
        if not np.isfinite(degrees) or not 0 <= degrees <= 10:
            raise ValueError('Tilt bias must be between 0 and 10 degrees')
        self.tilt_bias = float(degrees)

    def set_method(self, method):
        if method not in METHODS:
            raise ValueError(f'Drive method must be one of {METHODS}')
        self.method = method

    def reset(self, require_release=False):
        """Also after the plant resampled the robot, so the tilt applied so far is gone with it."""
        self.direction = 0
        self.offset = 0.0
        self.turn = 0.0
        self.speed = 0.0
        self.hold_creep = False
        self.tilt_applied = 0.0

    def update(self, direction, turn, wheel_rpm, wheel_radius):
        """Returns (common, differential) wheel commands; the plant sends common -+ differential to left/right."""
        self.direction = direction
        self.speed = float(np.mean(wheel_rpm) * 2 * np.pi / 60 * wheel_radius)
        parts = self.method.split('+')
        self.offset = float(-direction * self.command) if 'push' in parts else 0.0
        self.turn = float(turn * self.turn_command)
        uses_creep = 'push' in parts or 'creep' in parts
        self.hold_creep = uses_creep and (bool(direction) or (self.hold_creep and abs(self.speed) > self.STOPPED_MPS))
        return self.offset, self.turn

    def tilt_change(self):
        """Degrees to add to the IMU mounting tilt this step."""
        want = -self.direction * self.tilt_bias if 'tilt' in self.method.split('+') else 0.0
        change, self.tilt_applied = want - self.tilt_applied, want
        return change

    def creep(self):
        """Value (revolutions) to hold the creep integral at this step, or None to leave it alone."""
        if not self.hold_creep:
            return None
        return self.direction * self.creep_bias if 'creep' in self.method.split('+') else 0.0

    def state(self):
        return {'command': self.command, 'turn_command': self.turn_command,
                'creep_bias': self.creep_bias, 'tilt_bias': self.tilt_bias,
                'method': self.method, 'methods': list(METHODS),
                'offset': -self.offset, 'speed': self.speed,
                'status': 'Driving' if self.direction else 'Stopping' if self.hold_creep else
                          'Turning' if self.turn else 'Balancing'}
