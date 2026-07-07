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
    float dtSeconds
) {
    const float targetPitchDeg = targetLinear * MAX_LEAN_DEG;

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

    return {speed, steer};
}

void BalanceController::reset() {
    balancePid.reset();
}
