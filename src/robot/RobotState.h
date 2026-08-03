#pragma once

#include <Arduino.h>

struct RobotState {
    bool enabled = false;

    float targetLinear = 0.0f;
    float targetAngular = 0.0f;

    float pitchDeg = 0.0f;
    float pitchRateDegPerSec = 0.0f;

    // Live-tunable balance gains (initialized from config in setup(); SET_PID
    // changes them at runtime, lost on reboot).
    float balanceKp = 0.0f;
    float balanceKi = 0.0f;
    float balanceKd = 0.0f;

    float batteryVoltage = 0.0f;
    float leftWheelSpeed = 0.0f;
    float rightWheelSpeed = 0.0f;

    unsigned long lastCommandMs = 0;
    unsigned long lastImuUpdateMs = 0;
    unsigned long lastMotorFeedbackMs = 0;
};
