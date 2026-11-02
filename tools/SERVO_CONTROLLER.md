# Three-servo controller

Signals: servo 1 GPIO18, servo 2 GPIO19, servo 3 GPIO23. Use an external
supply rated for your servos and connect supply ground to ESP32 ground.
This controller is for positional hobby servos, not continuous-rotation servos.

## Setup and run

Upload the updated firmware using PlatformIO's Upload button, or from the repo root:

```sh
~/.platformio/penv/bin/pio run -t upload
```

For this Mac's Homebrew Python 3.13, install its missing GUI support:

```sh
brew install python-tk@3.13
python3 -m venv .venv-servo
.venv-servo/bin/python -m pip install pyserial
.venv-servo/bin/python tools/servo_controller.py
```

On other machines use a Python distribution with Tkinter and install pyserial
in a virtual environment. Wi-Fi mode does not require pyserial.

Choose USB, refresh ports, select the ESP32 port, and Connect. Close other serial
monitors/controllers first. Alternatively join BizBot-Control Wi-Fi and select
Wi-Fi with address 192.168.4.1:3333. An active Wi-Fi controller takes priority over USB.

Connecting releases all servos. Select an angle on a slider, then press that
servo's Move button. Slider motion alone does not send commands. The status line
shows the firmware's acknowledgement, not measured shaft position.

The firmware maps nominal 0–180 degrees to 1000–2000 microseconds at 50 Hz.
Actual shaft angle and safe travel depend on the servo and linkage. Adjust
SERVO_MIN_PULSE_US / SERVO_MAX_PULSE_US in src/config.h for your servo specification,
then rebuild/upload. Start near the midpoint with the linkage unloaded.

Release stops control pulses but does not disconnect servo power or guarantee
that every digital servo stops holding. Disconnecting requests release; if the
link fails, firmware stops pulses after two seconds without a servo command or
heartbeat. Support any load that could fall when holding torque is lost.
No servo pulses are generated at boot. This GUI never sends ENABLE or drive
commands, and servo heartbeats do not refresh the wheel controller's watchdog.
Do not run it alongside a wheel controller.

## Protocol

- `SERVO 1 90` — set servo 1's nominal angle (IDs 1–3, finite angles 0–180).
- `SERVO_OFF 1` — stop pulses for servo 1.
- `SERVO_OFF` — stop all servo pulses.
- `SERVO_PING` — refresh the two-second servo timeout.

PWM channels 0–2 are reserved for servos. Firmware uses the project's installed
Arduino ESP32 2.x LEDC API. USB baud rate is 115200.

## Verification

```sh
python3 -m unittest discover -s tools/tests
~/.platformio/penv/bin/pio run
```

Tests cover angle validation, fragmented replies, telemetry filtering,
unsupported firmware, and transport disconnects. Hardware angle calibration
and power delivery must be checked on the actual servos.

## Standalone sweep firmware

See [the independent servo-sweep project](../firmware/servo-sweep/README.md).
