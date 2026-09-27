# Acceleration damping

Built on the restored b613aca angle controller. This does not restore the
wheel-velocity controller and does not hold a fixed position or speed.

RVC acceleration (m/s²) is projected onto the horizontal sensor-Y heading using
the raw sensor roll/pitch, removing the gravity component under the documented
measured RVC convention in `AccelerationFeedback.h`. Balance trim is deliberately not
used for gravity compensation. A 100 ms first-order filter updates only when
new IMU packets arrive.

Side mounting: gravity is near sensor +X, roll near -90 degrees, and sensor Y
is treated as fore/aft. `IMU_FORWARD_ACCEL_SIGN` maps the result to positive robot-forward
acceleration. This mapping and the Euler convention require hardware
verification after remounting. An IMU away from the axle also measures motion
from chassis rotation; this is not removed by gravity compensation.

With balancing disabled, use GET_STATE to inspect `accel`. Hold the chassis
still at several pitch/roll angles and let the filter settle: acceleration
should be near zero at each angle. Then translate it forward without tilting:
the initial acceleration should be positive and braking should be negative.
Do not enable feedback if these checks fail. Constant-speed translation should
read approximately zero; that is expected.

The default gain is zero. SET_ACCEL rejects a nonzero gain unless IMU state is
fresh and absolute filtered acceleration is below 0.5 m/s². This guard does not
replace the mounting/direction checks. In the keyboard script, press T and enter:

* `accel 0.01` for an initial small experimental gain after mounting checks.
* `accel 0` to remove acceleration feedback.

The serial equivalents are `SET_ACCEL 0.01` and `SET_ACCEL 0`. Gains are limited
to 0–0.1 normalized motor output per m/s² and reset on reboot. Corrections oppose measured
acceleration, are limited to 0.03 normalized motor output, and fade to zero at
10 degrees of tilt. This correction is added after the balance PID motor sign
mapping; FORWARD_MOTOR_SIGN must match physical forward motor command direction.
Motor output limits and IMU/command fault handling remain in force.

Telemetry: `accel` is filtered horizontal acceleration (m/s²), `ka` is its gain,
and `acorr` is the opposing motor correction before FORWARD_MOTOR_SIGN. PID
`out` includes the correction; the angle target is unchanged.
Q disables balancing. ZERO_IMU sets the balance reference, not acceleration bias.
Offsets set with ZERO_IMU or SET_TRIM (keyboard `zero`, `offset`, or `trim`)
are saved to ESP32 NVS and restored after reboot and normal firmware uploads.
A full flash erase clears them. A failed save returns ERR OFFSET_SAVE_FAILED
and leaves the previous offset active. With no valid saved offset, the firmware
uses PITCH_TRIM_DEG from config.h. PID and acceleration gains remain session-only.

Software validation:

```sh
c++ -std=c++11 test/acceleration/test_feedback.cpp -o /tmp/bizbot-acceleration-test
/tmp/bizbot-acceleration-test
pio run
```

The synthetic tests cover static gravity under tilt, horizontal acceleration,
feedback sign, zero gain, bounded correction, recovery fade, and invalid input.
They do not establish the physical sensor mounting or closed-loop stability.
