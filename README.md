# bizbot-firmware

BizBot is a self-balancing robot: an ESP32 reads a BNO085 IMU and drives a
hoverboard mainboard (EFeru hoverboard-firmware-hack-FOC, USART) over UART.
This repo holds the ESP32 firmware, the MuJoCo sim and RL training that
produce its balance policy, the laptop tools for driving and tuning it, the PCB,
and the iPad face app.

```
Laptop --USB serial or Wi-Fi TCP--> ESP32 --UART--> hoverboard board --> wheels
                                      ^
                                BNO085 (UART-RVC)
```

## Common commands

Run `make` for the list. Override variables on the command line, e.g.
`make flash PORT=/dev/cu.usbserial-10`.

| Command | What it does |
| --- | --- |
| `make build` | Build the firmware in `firmware/` |
| `make flash` | Build and upload to `PORT`. **Turn the hoverboard off first.** If it is on while flashing, its UART stalls |
| `make servo-sweep` | Upload the separate servo test in `firmware/servo-sweep/` |
| `make drive` | Join `BizBot-Control` Wi-Fi first. Opens the drive page (`tools/drive_ui.py`) unless a tab is already open: W/S sets the target wheel speed, A/D steers. It balances on connect and logs to `~/bizbot_<time>.log` |
| `make tuner` | Browser tuner at http://localhost:5173 (`tools/tuner/`) |
| `make sim` | MuJoCo viewer running `POLICY` on `ROBOT` |
| `make export` | Export `POLICY` to C in `OUT` (default `model/deploy`) and run the host test |
| `make test` | Sim/drive unit tests in `model/mj/tests` |

The Makefile uses PlatformIO from `~/.platformio/penv/bin/pio` and Python from
`.venv/bin/python`. Laptop tool dependencies are in `requirements.txt`, the sim
needs `sim/requirements.txt`, and the tuner needs `npm install` in `tools/tuner/`.

## Firmware

`firmware/` is a PlatformIO project (`esp32dev`, Arduino, Adafruit BNO08x RVC).

```
firmware/src/
├── main.cpp            setup/loop, Wi-Fi AP, hoverboard UART, 5 ms control loop, safety stops
├── config.h            pins, limits, gains, IMU mounting, build switches
├── hal/                Imu (BNO085 UART-RVC), OffsetStorage (saved trim), ServoControl
├── control/            PID, BalanceController, VelocityControl, VelocityEstimator,
│                       AccelerationFeedback, bizbot_policy.c/.h (generated RL policy)
├── comms/              CommandParser (line protocol below)
└── robot/RobotState.h  shared state
```

### Balance controller

When enabled, the **RL policy** drives the wheels. It is generated C in
`firmware/src/control/bizbot_policy.c`, currently from `fit_tgt_E/final.pt`, and runs
every second 5 ms tick. The PID balance controller still runs every tick, but
only to compute steering and to be logged. Its speed command is not sent. Each
policy tick emits one line, over Wi-Fi when a client is connected and over USB
otherwise:

```
RL <ms> <pitch_deg> <accel_fwd> <wheel_rpm> <battery_v> <pid_cmd> <rl_cmd> <us>
```

The robot stops and disables on Wi-Fi disconnect, an IMU fault or stale data,
a command timeout (500 ms; `drive_ui.py` keeps alive every 100 ms, `keyboard_controls.py` every 200 ms),
a tilt beyond 35°, or a control-loop overrun. Each stop prints `FAULT <reason>`
on USB.

### Build switches (`config.h`, or `build_flags` in `platformio.ini`)

- `IMU_USE_STUB=1`: no IMU wired; commands work but motor output is forced to zero.
- `IMU_DEBUG_EULER=1`: stream raw Euler angles for mounting calibration or
  `tools/imu_visualizer.py`.
- `IMU_UART_RAW_DUMP=1`: hex-dump the IMU UART at boot.
- `PID_DEBUG=1`: print controller internals.
- `MOTOR_TEST=1`: send a constant speed command.

All default to 0. IMU wiring and calibration are in `docs/IMU_TESTING.md`.

### Command protocol

Commands go over USB serial (115200 baud) or Wi-Fi TCP. Each one is a line
ending in `\n` and gets an `OK ...`, `ERR ...` or `STATE ...` reply.

```text
ENABLE | DISABLE | STOP | GET_STATE
CMD_VEL <linear> <angular>          normalized -1.0..1.0; requires ENABLE
DRIVE <target_rpm> <steer>          RL target wheel speed, ±150 rpm, + forward; steer ±200
                                    hoverboard units; requires ENABLE; STOP zeroes both
SET_PID <kp> <ki> <kd>
SET_PID_FB <kp_front> <kp_back> <ki> <kd>
SET_P_BOOST <boost>   SET_P_WIDTH <deg>
SET_D_BOOST <boost> <width_deg>
SET_ACCEL <gain>                    see docs/ACCELERATION_FEEDBACK.md
SET_VEL <kp> <ki>     SET_VEL_SIGN <±1>
SET_FUSION <wheel_diameter_mm> <tau>
SET_TRIM <deg>        ZERO_IMU      trim is saved to flash
SERVO <1-3> <0-180>   SERVO_OFF [1-3]   SERVO_PING
```

The gain commands tune the PID, which is not what drives the wheels while the RL
policy is in charge (see above).

### Wi-Fi

The ESP32 runs its own access point. It does not need a router or an internet connection.

```text
SSID: BizBot-Control   Password: bizbot-control   (change in firmware/src/config.h)
ESP32: 192.168.4.1     TCP control port: 3333
```

It accepts one TCP controller at a time. While that controller is connected it
owns command input, and USB is only a debug console. When it disconnects, the
robot is disabled and USB command input comes back.

### Host tests

From `firmware/`:

```bash
c++ -std=c++11 -I test/pid src/control/PID.cpp test/pid/test_derivative_boost.cpp -o /tmp/bizbot-pid-test && /tmp/bizbot-pid-test
c++ -std=c++11 -Isrc test/velocity/check.cpp -o /tmp/bizbot-velocity-check && /tmp/bizbot-velocity-check
```

## Balance policy pipeline

- `sim/bizbot_mj/`: MuJoCo plant and vectorised balance env, with
  domain-randomised robot parameters. `sim/bizbot_sim/` is the older numpy sim.
- `model/mj/`: training (`ppo.py`, `td3.py`, `pbt.py`, `distill.py`, `sweep.py`)
  and evaluation (`metrics.py`, `evalckpt.py`, `baselines.py`, `summarize.py`).
  It also has `watch.py` (viewer), `export_policy.py`, and tests in `tests/`.
  Runs and checkpoints go to `model/mj/runs/` (gitignored).
- `model/deploy/`: output of `make export`, which writes `bizbot_policy.c/.h/.json`,
  `policy.pt`, `test_policy.c`, `test_vectors.h` and `SPEC.md` (the I/O contract).
  To deploy, copy `bizbot_policy.c/.h` into `firmware/src/control/`. The copy in
  `model/deploy/` is currently from `ckpt_0040M.pt`, older than the firmware's.

## Other directories

- `tools/`: `drive_ui.py` (browser drive page, serial or Wi-Fi), `keyboard_controls.py` (terminal W/A/S/D), `tuner/` (browser
  tuner that spawns `keyboard_controls.py`), `servo_controller.py` (see
  `SERVO_CONTROLLER.md`), `imu_visualizer.py`, and `tests/`, which need a
  Python with Tkinter.
- `firmware/servo-sweep/`: a separate PlatformIO project that sweeps three servos.
  It runs no BizBot firmware.
- `hardware/`: KiCad schematic, PCB and Gerber exports.
- `interaction/`: native iPad face app, its Swift core and tests, and a Node
  server. It sends no motor commands. See `interaction/README.md`.
- `docs/`: `IMU_TESTING.md` and `ACCELERATION_FEEDBACK.md`.
