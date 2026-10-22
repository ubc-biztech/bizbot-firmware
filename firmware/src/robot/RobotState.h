#pragma once

#include <Arduino.h>

struct RobotState {
    bool enabled = false;

    float targetLinear = 0.0f;
    float targetAngular = 0.0f;

    float pitchDeg = 0.0f;
    float pitchRateDegPerSec = 0.0f;
    float forwardAccel = 0.0f;
    float accelGain = 0.0f;
    float accelCorrection = 0.0f;
    float rawAccelX = 0, rawAccelY = 0, rawAccelZ = 0;
    float rawRoll = 0, rawPitch = 0;

    // Live-tunable balance gains (initialized from config in setup(); SET_PID
    // changes them at runtime, lost on reboot).
    float balanceKp = 0.0f;
    float balanceKi = 0.0f;
    float balanceKd = 0.0f;

    // Outer velocity loop (SET_VEL / SET_VEL_SIGN), see config.h.
    float velKp = 0.0f;
    float velKi = 0.0f;
    float velSign = -1.0f;
    float wheelVelocity = 0.0f;      // mean wheel RPM, forward positive
    float velIntegral = 0.0f;        // integrated velocity error
    float targetPitchDeg = 0.0f;     // what the angle loop is tracking

    // Persistent balance-point offset (SET_TRIM / ZERO_IMU), loaded from NVS.
    float pitchTrimDeg = 0.0f;
    bool resetBalanceRequested = false;

    float batteryVoltage = 0.0f;
    float leftWheelSpeed = 0.0f;
    float rightWheelSpeed = 0.0f;

    unsigned long lastCommandMs = 0;
    unsigned long lastImuUpdateMs = 0;
    unsigned long lastMotorFeedbackMs = 0;
};
