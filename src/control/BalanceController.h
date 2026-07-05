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
        float pitchRateDegPerSec,
        float dtSeconds
    );

    void reset();

private:
    PID balancePid;
};
