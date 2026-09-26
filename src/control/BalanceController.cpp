#include "BalanceController.h"

#include "../config.h"

#include <Arduino.h>

BalanceController::BalanceController(float kp, float ki, float kd)
    : balancePid(kp, ki, kd) {
    balancePid.setOutputLimits(
        -MAX_BALANCE_OUTPUT,
        MAX_BALANCE_OUTPUT
    );
}

BalanceOutput BalanceController::update(
    float targetLinear,
    float targetAngular,
    float pitchDeg,
    float pitchRateDegPerSec,
    float dtSeconds,
    float velocityAngleCorrectionDeg
) {
    // pitchDeg has the calibrated trim subtracted in main.cpp. Therefore this
    // relative target is equivalent to trim + manual lean + velocity correction
    // in the IMU frame. Preserve the existing manual lean command and mixing.
    const float targetPitchDeg =
        targetLinear * MAX_LEAN_DEG + velocityAngleCorrectionDeg;

    const float speed = balancePid.update(
        targetPitchDeg,
        pitchDeg,
        dtSeconds,
        pitchRateDegPerSec
    );

    const float steer = constrain(
        targetAngular * MAX_TURN_OUTPUT,
        -MAX_TURN_OUTPUT,
        MAX_TURN_OUTPUT
    );

    const float unclampedSpeed = BALANCE_MOTOR_SIGN *
        (balancePid.lastPTerm() + balancePid.lastITerm() + balancePid.lastDTerm());
    return {BALANCE_MOTOR_SIGN * speed, steer, unclampedSpeed};
}

void BalanceController::setTunings(float kp, float ki, float kd) {
    balancePid.setTunings(kp, ki, kd);
}

void BalanceController::reset() {
    balancePid.reset();
}
