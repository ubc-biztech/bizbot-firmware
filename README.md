# bizbot-firmware
This firmware controls the low-level robot layer for BizBot.

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