# Cascaded stationary velocity control

Previously, `loop()` processed commands and wheel feedback, then ran the angle
PID every 5 ms. The PID compared `targetLinear * MAX_LEAN_DEG` against
`IMU pitch - pitchTrimDeg`, with derivative on measured pitch rate. Wheel
velocities were only reported. Speed was limited to ±0.20, multiplied by
`BALANCE_MOTOR_SIGN`, and sent with the independently limited steering command.
This maintained angle without feeding rolling speed back into the target angle.

Now a 25 Hz P controller samples the existing wheel feedback. Between samples,
the 200 Hz angle loop holds its angle correction:

```text
left    = LEFT_WHEEL_VELOCITY_SIGN  * speedL_meas
right   = RIGHT_WHEEL_VELOCITY_SIGN * speedR_meas
forward = (left + right) / 2
error   = 0 - forward
correction = clamp(VELOCITY_ANGLE_SIGN * VELOCITY_KP * error, -5°, +5°)
absolute target = pitchTrimDeg + targetLinear * MAX_LEAN_DEG + correction
```

The angle PID still uses trim-relative pitch, so only the manual lean plus
correction is passed as its relative target. Trim is not added twice. Positive
pitch means forward lean: positive forward speed produces a negative correction
(backward target lean), and negative forward speed produces a positive correction.
The angle PID, motor polarity, steering mapping, ±0.20 speed limit, ±0.15 steering
limit, command timeout, IMU protection and 35° tilt cutoff are retained.

The target velocity is always zero in this stationary-balancing implementation.
The existing `CMD_VEL` linear field remains a manual lean bias; it has not been
reinterpreted as a physical velocity. Use linear=0 for stationary shove tests.
A nonzero manual lean therefore competes with the zero-speed feedback. True
commanded velocity tracking needs an explicitly calibrated command-to-speed scale.

## Calibration constants and uncertainties

All settings are in `src/config.h`. `VELOCITY_KP=0.01` degrees per raw feedback unit
is an initial, unvalidated tuning value; `VELOCITY_KI=VELOCITY_KD=0`. There is no
velocity integral state or derivative term. A compile-time assertion prevents
silently enabling unimplemented I/D gains. Future integral support must include
anti-windup. `MAX_VELOCITY_LEAN_DEG=5` is the correction limit.

The repository does not establish the physical velocity units or wheel polarity.
No RPM or metres/second conversion is assumed. Both wheel-sign defaults are
provisional +1. With the robot disabled, push it forward and inspect telemetry.
Set each wheel's sign independently to ±1 so both normalized readings are positive
when moving forward; backward motion must make both negative. Verify this before
enabling the outer loop on the physical robot. Opposite raw wheel signs can
otherwise cancel in the average. Keep `VELOCITY_ANGLE_SIGN=+1` for the documented
positive-forward pitch convention; it is separate from the motor-output sign.

Missing feedback or feedback older than `MOTOR_FEEDBACK_TIMEOUT_MS=250` stops the
robot with `FAULT MOTOR_FEEDBACK`. This additional check is needed because the
outer loop now depends on wheel measurements. Confirm the board's feedback period
fits this timeout. Disable, re-enable and IMU zeroing clear the held correction.

## Telemetry and timing

With `PID_DEBUG=1` (enabled in the current PlatformIO configuration), `CTRL` lines
are emitted at 10 Hz over USB, including while Wi-Fi owns command input. TCP is not
used for telemetry because it can block. Lines are dropped if the USB TX buffer
has insufficient space. Motor UART writes also check buffer space. Fault messages
are deferred outside `runControlLoop`; IMU polling, including optional Euler
logging, takes place outside that function. The scheduler remains cooperative;
these changes do not provide a hard real-time guarantee under command/network load.

Fields:

- `wheel_l`, `wheel_r`: normalized wheel velocities, in raw board units.
- `forward`, `vel_error`: wheel average and zero-target error, in the same units.
- `angle_corr`: held correction in degrees (zero when inactive).
- `target`, `pitch`: final target and measured pitch in the same absolute,
  mounting-normalized IMU frame, including the calibrated trim.
- `pitch_rate_derived`: degrees/second from the existing filtered angle derivative.
  **UART-RVC provides no gyro rate**, so this is not a gyroscope measurement.
- `motor_raw`, `motor_clamped`: normalized speed output before/after the existing
  balance clamp, both including motor polarity. The hoverboard command is the
  clamped value multiplied by `MAX_HOVERBOARD_COMMAND` and converted to int16.
- `enabled`, `feedback_ok`: control and wheel-feedback validity indicators.

While enabled, velocity fields reflect the exact outer-loop sample held for
control. While disabled, they show the latest wheels for sign calibration.

## Validation

Build firmware with `pio run`. Run the host regression tests from the repo root:

```sh
g++ -std=c++11 -Wall -Wextra -Werror -Itest/control -Isrc \
  test/control/test_cascade.cpp src/control/BalanceController.cpp \
  src/control/PID.cpp -o /tmp/bizbot-test-cascade
/tmp/bizbot-test-cascade
```

Tests exercise normalization/averaging, both braking directions, pure turning,
correction saturation, absence of residual integral, original balance output
with zero correction, preserved steering/speed limits, and pre-clamp telemetry.
They do not establish physical wheel signs, gain stability, or hardware timing.
No firmware upload or physical shove test is performed by these checks.
