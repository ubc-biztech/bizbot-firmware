# BizBot Firmware Handover

## Goal

Phase 1 firmware should:

1. Balance the robot in place.
2. Receive movement commands from a laptop.
3. Drive forward, backward, and turn.
4. Stop safely when commands or sensor data become invalid.

```text
Laptop
  |
  | USB serial: ENABLE, STOP, CMD_VEL, GET_STATE
  v
ESP32
  |
  | UART binary protocol
  v
Hoverboard controller
  |
  v
Motors
```

The ESP32 owns IMU sampling, balancing, motor control, hardware feedback, and
safety. Camera, audio, LLM, dashboard, and other high-level features belong on
the laptop or future onboard computer.

## Current State

Implemented:

- PlatformIO project using Arduino and `esp32dev`.
- Shared `RobotState`.
- Non-blocking `CommandParser`.
- `ENABLE`, `DISABLE`, `STOP`, `CMD_VEL`, and `GET_STATE`.
- Python keyboard control script.
- Hoverboard UART send/receive code in `main.cpp`.

Still required:

- IMU hardware abstraction.
- Fixed-rate control loop.
- PID implementation.
- Balance controller.
- Safety and command timeout handling.
- Mapping `RobotState` into hoverboard speed and steer commands.
- Encoder/wheel-speed velocity controller.

## Command Protocol

Commands are sent over USB serial at 115200 baud and end with `\n`.

```text
ENABLE
DISABLE
STOP
CMD_VEL <linear> <angular>
GET_STATE
```

`linear` and `angular` are normalized to `[-1.0, 1.0]`.

```text
ENABLE
CMD_VEL 0.20 0.00
CMD_VEL 0.00 -0.20
GET_STATE
STOP
DISABLE
```

`CommandParser` only updates `RobotState`. It should not call the motor driver
directly.

## Next Steps

### 1. Add the IMU Module

Create:

```text
src/hal/Imu.h
src/hal/Imu.cpp
```

`src/hal/Imu.h`:

```cpp
#pragma once

class Imu {
public:
    bool begin();
    bool update();

    float getPitchDeg() const;
    float getPitchRateDegPerSec() const;

private:
    float pitchDeg = 0.0f;
    float pitchRateDegPerSec = 0.0f;
};
```

`src/hal/Imu.cpp`:

```cpp
#include "Imu.h"
#include <Arduino.h>
#include <Wire.h>

bool Imu::begin() {
    Wire.begin();

    // Initialize the specific IMU library here.
    // Configure its range, sample rate, and filters.
    // Return false if the sensor cannot be detected.

    return true;
}

bool Imu::update() {
    // Read the specific IMU here.
    // Convert its mounted axes into robot pitch and pitch rate.
    //
    // pitchDeg = ...;
    // pitchRateDegPerSec = ...;

    return true;
}

float Imu::getPitchDeg() const {
    return pitchDeg;
}

float Imu::getPitchRateDegPerSec() const {
    return pitchRateDegPerSec;
}
```

The implementation depends on the exact IMU model. Record its:

```text
model
I2C address
SDA/SCL pins
mounting direction
pitch axis and sign
PlatformIO library
```

Update `RobotState`:

```cpp
struct RobotState {
    bool enabled = false;

    float targetLinear = 0.0f;
    float targetAngular = 0.0f;

    float pitchDeg = 0.0f;
    float pitchRateDegPerSec = 0.0f;
    float batteryVoltage = 0.0f;

    unsigned long lastCommandMs = 0;
    unsigned long lastImuUpdateMs = 0;
};
```

Use it from the control loop:

```cpp
if (!imu.update()) {
    robotState.enabled = false;
    Send(0, 0);
    return;
}

robotState.pitchDeg = imu.getPitchDeg();
robotState.pitchRateDegPerSec = imu.getPitchRateDegPerSec();
robotState.lastImuUpdateMs = millis();
```

### 2. Add the PID Module

Create:

```text
src/control/PID.h
src/control/PID.cpp
```

`src/control/PID.h`:

```cpp
#pragma once

class PID {
public:
    PID(float kp, float ki, float kd);

    float update(float target, float measurement, float dtSeconds);
    void setTunings(float kp, float ki, float kd);
    void reset();

private:
    float kp;
    float ki;
    float kd;

    float integral = 0.0f;
    float previousError = 0.0f;
    bool firstUpdate = true;
};
```

`src/control/PID.cpp`:

```cpp
#include "PID.h"

PID::PID(float kp, float ki, float kd)
    : kp(kp), ki(ki), kd(kd) {}

float PID::update(
    float target,
    float measurement,
    float dtSeconds
) {
    const float error = target - measurement;

    integral += error * dtSeconds;

    float derivative = 0.0f;
    if (!firstUpdate && dtSeconds > 0.0f) {
        derivative = (error - previousError) / dtSeconds;
    }

    previousError = error;
    firstUpdate = false;

    return (kp * error) + (ki * integral) + (kd * derivative);
}

void PID::setTunings(float newKp, float newKi, float newKd) {
    kp = newKp;
    ki = newKi;
    kd = newKd;
}

void PID::reset() {
    integral = 0.0f;
    previousError = 0.0f;
    firstUpdate = true;
}
```

Initial construction:

```cpp
PID balancePid(
    0.04f,   // Kp placeholder
    0.0f,    // Ki starts at zero
    0.001f   // Kd placeholder
);
```

The values are placeholders and must be tuned on the physical robot.

### 3. Add the Balance Controller

Create:

```text
src/control/BalanceController.h
src/control/BalanceController.cpp
```

`src/control/BalanceController.h`:

```cpp
#pragma once

#include "PID.h"

struct BalanceOutput {
    float speed;
    float steer;
};

class BalanceController {
public:
    BalanceController(float kp, float ki, float kd);

    BalanceOutput update(
        float targetLinear,
        float targetAngular,
        float pitchDeg,
        float dtSeconds
    );

    void reset();

private:
    PID balancePid;
};
```

`src/control/BalanceController.cpp`:

```cpp
#include "BalanceController.h"
#include <Arduino.h>

namespace {
constexpr float MAX_LEAN_DEG = 5.0f;
constexpr float MAX_BALANCE_OUTPUT = 0.20f;
constexpr float MAX_TURN_OUTPUT = 0.15f;
}

BalanceController::BalanceController(float kp, float ki, float kd)
    : balancePid(kp, ki, kd) {}

BalanceOutput BalanceController::update(
    float targetLinear,
    float targetAngular,
    float pitchDeg,
    float dtSeconds
) {
    const float targetPitchDeg =
        targetLinear * MAX_LEAN_DEG;

    float speed = balancePid.update(
        targetPitchDeg,
        pitchDeg,
        dtSeconds
    );

    speed = constrain(
        speed,
        -MAX_BALANCE_OUTPUT,
        MAX_BALANCE_OUTPUT
    );

    const float steer = constrain(
        targetAngular * MAX_TURN_OUTPUT,
        -MAX_TURN_OUTPUT,
        MAX_TURN_OUTPUT
    );

    return {speed, steer};
}

void BalanceController::reset() {
    balancePid.reset();
}
```

For the first balance test, ignore movement commands:

```cpp
BalanceOutput output = balanceController.update(
    0.0f,
    0.0f,
    robotState.pitchDeg,
    CONTROL_DT_SECONDS
);
```

After balancing works, pass the actual state:

```cpp
BalanceOutput output = balanceController.update(
    robotState.targetLinear,
    robotState.targetAngular,
    robotState.pitchDeg,
    CONTROL_DT_SECONDS
);
```

### 4. Add the Fixed-Rate Control Loop

Run the IMU and balance controller at 200 Hz:

```cpp
constexpr uint32_t CONTROL_PERIOD_US = 5000;
constexpr float CONTROL_DT_SECONDS = 0.005f;

uint32_t lastControlUs = 0;
```

Add:

```cpp
void runControlLoop() {
    if (!imu.update()) {
        robotState.enabled = false;
        balanceController.reset();
        Send(0, 0);
        DebugSerial.println("FAULT IMU");
        return;
    }

    robotState.pitchDeg = imu.getPitchDeg();
    robotState.pitchRateDegPerSec =
        imu.getPitchRateDegPerSec();
    robotState.lastImuUpdateMs = millis();

    if (!robotState.enabled) {
        balanceController.reset();
        Send(0, 0);
        return;
    }

    BalanceOutput output = balanceController.update(
        0.0f,  // use robotState.targetLinear after balance works
        0.0f,  // use robotState.targetAngular after balance works
        robotState.pitchDeg,
        CONTROL_DT_SECONDS
    );

    const int16_t speed = normalizedToHoverboard(output.speed);
    const int16_t steer = normalizedToHoverboard(output.steer);

    Send(steer, speed);
}
```

Schedule it from `loop()`:

```cpp
void loop() {
    commandParser.update(
        DebugSerial,
        DebugSerial,
        robotState
    );

    Receive();

    const uint32_t now = micros();
    if (now - lastControlUs >= CONTROL_PERIOD_US) {
        lastControlUs += CONTROL_PERIOD_US;
        runControlLoop();
    }
}
```

Do not use `delay()` or blocking serial reads in this loop.

### 5. Apply Robot State in `main.cpp`

Add the module includes and global objects:

```cpp
#include "comms/CommandParser.h"
#include "control/BalanceController.h"
#include "hal/Imu.h"
#include "robot/RobotState.h"

HardwareSerial& HoverSerial = Serial2;
HardwareSerial& DebugSerial = Serial;

RobotState robotState;
CommandParser commandParser;
Imu imu;
BalanceController balanceController(
    0.04f,
    0.0f,
    0.001f
);
```

Initialize everything in `setup()`:

```cpp
void setup() {
    DebugSerial.begin(115200);

    HoverSerial.begin(
        HOVER_SERIAL_BAUD,
        SERIAL_8N1,
        HOVER_RX_PIN,
        HOVER_TX_PIN
    );

    if (!imu.begin()) {
        robotState.enabled = false;
        DebugSerial.println("FAULT IMU_INIT");
    }

    Send(0, 0);
}
```

Convert normalized controller output into the hoverboard command range:

```cpp
constexpr int16_t MAX_HOVERBOARD_COMMAND = 50;

int16_t normalizedToHoverboard(float value) {
    value = constrain(value, -1.0f, 1.0f);

    return static_cast<int16_t>(
        value * MAX_HOVERBOARD_COMMAND
    );
}
```

`MAX_HOVERBOARD_COMMAND` must be confirmed using restrained, low-power
hardware tests.

Update `RobotState` from hoverboard feedback:

```cpp
if (
    NewFeedback.start == START_FRAME &&
    checksum == NewFeedback.checksum
) {
    Feedback = NewFeedback;

    robotState.batteryVoltage =
        Feedback.batVoltage / 100.0f;

    robotState.leftWheelSpeed =
        Feedback.speedL_meas;

    robotState.rightWheelSpeed =
        Feedback.speedR_meas;
}
```

Add these fields:

```cpp
float leftWheelSpeed = 0.0f;
float rightWheelSpeed = 0.0f;
```

### 6. Add Safety Handling

Add limits:

```cpp
constexpr unsigned long COMMAND_TIMEOUT_MS = 500;
constexpr float MAX_TILT_DEG = 35.0f;
```

Add:

```cpp
bool commandTimedOut() {
    return millis() - robotState.lastCommandMs >
        COMMAND_TIMEOUT_MS;
}

bool tiltLimitExceeded() {
    return fabs(robotState.pitchDeg) > MAX_TILT_DEG;
}

void stopRobot(const char* reason) {
    robotState.enabled = false;
    robotState.targetLinear = 0.0f;
    robotState.targetAngular = 0.0f;

    balanceController.reset();
    Send(0, 0);

    DebugSerial.print("FAULT ");
    DebugSerial.println(reason);
}
```

Use the checks before applying motor output:

```cpp
if (commandTimedOut()) {
    stopRobot("COMMAND_TIMEOUT");
    return;
}

if (tiltLimitExceeded()) {
    stopRobot("TILT_LIMIT");
    return;
}
```

At startup, after reset, when disabled, and after any fault:

```cpp
Send(0, 0);
```

### 7. Add Velocity Control Later

After balancing and basic movement work, use wheel-speed feedback for an outer
velocity loop:

```text
robotState.targetLinear
  -> velocity PID
  -> target pitch
  -> balance PID
  -> hoverboard speed command
```

Suggested module:

```text
src/control/VelocityController.h
src/control/VelocityController.cpp
```

Concept:

```cpp
const float measuredVelocity =
    (robotState.leftWheelSpeed +
     robotState.rightWheelSpeed) * 0.5f;

const float targetPitchDeg = velocityPid.update(
    robotState.targetLinear,
    measuredVelocity,
    VELOCITY_DT_SECONDS
);

const float speedOutput = balancePid.update(
    targetPitchDeg,
    robotState.pitchDeg,
    CONTROL_DT_SECONDS
);
```

## Recommended Robot State

Add fields as their modules are implemented:

```cpp
#pragma once

#include <Arduino.h>

struct RobotState {
    bool enabled = false;

    float targetLinear = 0.0f;
    float targetAngular = 0.0f;

    float pitchDeg = 0.0f;
    float pitchRateDegPerSec = 0.0f;

    float batteryVoltage = 0.0f;
    float leftWheelSpeed = 0.0f;
    float rightWheelSpeed = 0.0f;

    unsigned long lastCommandMs = 0;
    unsigned long lastImuUpdateMs = 0;
    unsigned long lastMotorFeedbackMs = 0;
};
```

An explicit mode and fault enum can replace `enabled` after basic balancing is
working:

```cpp
enum class RobotMode {
    Disabled,
    Balancing,
    Fault
};

enum class FaultCode {
    None,
    CommandTimeout,
    ImuFailure,
    TiltLimit,
    MotorCommunication
};
```

## Implementation Order

1. Record the exact IMU model, pins, mounting orientation, and library.
2. Implement `Imu.h` and `Imu.cpp`.
3. Add IMU fields to `RobotState`.
4. Implement `PID.h` and `PID.cpp`.
5. Implement `BalanceController.h` and `BalanceController.cpp`.
6. Add the fixed-rate control loop to `main.cpp`.
7. Run the balance calculation with speed and steer forced to zero.
8. Add command timeout, tilt limit, and IMU fault handling.
9. Enable constrained low-power balance output.
10. Pass `targetLinear` and `targetAngular` into `BalanceController`.
11. Add wheel-speed velocity control.

## Hardware Information Required

- Exact ESP32 board model.
- Exact IMU model and breakout board.
- IMU voltage and logic-level requirements.
- I2C/SPI pins and address.
- IMU mounting direction.
- Hoverboard firmware version and UART baud rate.
- Confirmed hoverboard speed and steer ranges.
- Battery voltage range and low-voltage threshold.
- Physical emergency-stop method.

## Safety Rules

- Never power motors directly from the ESP32.
- Never connect the hoverboard 15 V wire to the ESP32.
- Share ground between the ESP32 and hoverboard controller.
- Keep wheels off the ground for initial direction tests.
- Restrain the chassis for initial balance tests.
- Start every boot and reset with zero motor output.
- Treat stale commands, stale sensor data, invalid feedback, and excessive
  tilt as stop conditions.
- Do not tune PID with unrestricted motor output.
