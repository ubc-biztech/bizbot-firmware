# bizbot-firmware
This firmware controls the low-level robot layer for BizBot.

# BizBot ESP32 Firmware

This firmware controls the low-level robot layer for BizBot.

## Responsibilities

- Read IMU
- Read encoders
- Control motors
- Run balance PID loop
- Receive high-level commands over USB serial or local Wi-Fi
- Send telemetry to host
- Enforce safety limits

## Current Prototype Architecture

Laptop -> USB Serial or Wi-Fi TCP -> ESP32 -> Motor Driver -> Motors

## Command Protocol

Send commands to the ESP32 over USB serial at 115200 baud or Wi-Fi TCP port
3333. Each command must end with a newline (`\n`).

```text
ENABLE
DISABLE
STOP
CMD_VEL <linear> <angular>
GET_STATE
```

`linear` and `angular` accept normalized values from `-1.0` to `1.0`.

```text
ENABLE
CMD_VEL 0.2 0.0
GET_STATE
STOP
DISABLE
```

## Local Wi-Fi control

The ESP32 creates its own 2.4 GHz access point; a router or internet connection
is not required.

```text
SSID: BizBot-Control
Password: bizbot-control
ESP32 address: 192.168.4.1
TCP control port: 3333
```

Change the prototype password in `src/config.h` before operating in public.
Only one TCP controller is accepted. While it is connected, Wi-Fi owns the
command input and USB remains a debug console. A Wi-Fi disconnect immediately
disables the robot and restores USB command input.

### Connect and test

1. Keep the wheels off the ground and leave the motor power disconnected for
   the initial communications test.
2. Install the host dependencies:

   ```bash
   python3 -m pip install -r requirements.txt
   ```

3. Build and upload using the PlatformIO IDE, or its CLI:

   ```bash
   pio run --target upload
   ```

4. Open the USB serial monitor at 115200 baud. On a successful boot it prints:

   ```text
   WiFi AP: BizBot-Control
   WiFi control: 192.168.4.1:3333
   ```

5. On the laptop, join the `BizBot-Control` Wi-Fi network using the password
   above. The laptop may report that this network has no internet; that is
   expected.
6. Start Wi-Fi keyboard control:

   ```bash
   python3 tools/keyboard_controls.py --transport wifi
   ```

   Use `W/A/S/D`, release the keys to stop, press `I` for state, and `Q` to
   stop, disable, and disconnect.

USB control remains available when no Wi-Fi TCP client is connected:

```bash
python3 tools/keyboard_controls.py --transport serial
```

The current PlatformIO configuration builds with `IMU_USE_STUB=1`. Stub builds
accept commands for communications testing but always force the hoverboard
output to zero. Do not set `IMU_USE_STUB=0` until the real IMU implementation
returns correctly oriented pitch and pitch-rate measurements. After installing
the real IMU, perform the first motor test with the chassis restrained and tune
the placeholder controller values before putting the wheels on the ground.

## Repo structure

```
bizbot-firmware/
├── platformio.ini
├── src/
│   ├── main.cpp
│   ├── config.h
│   │
│   ├── hal/
│   │   ├── Imu.h
│   │   ├── Imu.cpp
│   │   ├── MotorDriver.h
│   │   ├── MotorDriver.cpp
│   │   ├── Encoder.h
│   │   ├── Encoder.cpp
│   │   ├── Battery.h
│   │   └── Battery.cpp
│   │
│   ├── control/
│   │   ├── PID.h
│   │   ├── PID.cpp
│   │   ├── BalanceController.h
│   │   ├── BalanceController.cpp
│   │   ├── VelocityController.h
│   │   └── VelocityController.cpp
│   │
│   ├── comms/
│   │   ├── CommandParser.h
│   │   ├── CommandParser.cpp
│   │   ├── Telemetry.h
│   │   └── Telemetry.cpp
│   │
│   └── robot/
│       ├── RobotState.h
│       └── RobotState.cpp
│
├── tools/
│   └── keyboard_control.py
│
└── README.md
```

```
main.cpp
  runs setup()
  runs loop()
  calls everything else

hal/
  talks directly to hardware

control/
  computes motor outputs

comms/
  parses laptop commands and sends telemetry

robot/
  shared state: enabled, target velocity, current angle, battery, etc.

tools/
  Python scripts that run on laptop for testing
```
