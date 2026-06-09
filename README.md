# bizbot-firmware
This firmware controls the low-level robot layer for BizBot.

# BizBot ESP32 Firmware

This firmware controls the low-level robot layer for BizBot.

## Responsibilities

- Read IMU
- Read encoders
- Control motors
- Run balance PID loop
- Receive high-level commands over serial
- Send telemetry to host
- Enforce safety limits

## Current Prototype Architecture

Laptop -> USB Serial -> ESP32 -> Motor Driver -> Motors

## Command Protocol

Send commands to the ESP32 over USB serial at 115200 baud. Each command must end with a newline (`\n`).

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