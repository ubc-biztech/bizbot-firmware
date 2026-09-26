#pragma once

#include "PID.h"

struct BalanceOutput {
    float speed;
    float steer;
    float unclampedSpeed; // normalized motor output, including BALANCE_MOTOR_SIGN
};

class BalanceController {
public:
    BalanceController(float kp, float ki, float kd);

    BalanceOutput update(
        float targetLinear,
        float targetAngular,
        float pitchDeg,
        float pitchRateDegPerSec,
        float dtSeconds,
        float velocityAngleCorrectionDeg = 0.0f
    );

    void setTunings(float kp, float ki, float kd);
    void reset();

    // Last balance PID terms, for tuning diagnostics.
    float lastPTerm() const { return balancePid.lastPTerm(); }
    float lastITerm() const { return balancePid.lastITerm(); }
    float lastDTerm() const { return balancePid.lastDTerm(); }

private:
    PID balancePid;
};
