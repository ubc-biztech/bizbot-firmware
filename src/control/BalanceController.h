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
        float targetPitchDeg,
        float targetAngular,
        float pitchDeg,
        float pitchRateDegPerSec,
        float dtSeconds
    );

    void setTunings(float kp, float ki, float kd);
    void setProportionalBoost(float boost, float width) {
        balancePid.setProportionalBoost(boost, width);
    }
    void setDirectionalP(float front, float back) {
        balancePid.setDirectionalP(front, back);
    }
    void setDerivativeBoost(float boost, float width) {
        balancePid.setDerivativeBoost(boost, width);
    }
    void reset();

    // Last balance PID terms, for tuning diagnostics.
    float lastPTerm() const { return balancePid.lastPTerm(); }
    float lastITerm() const { return balancePid.lastITerm(); }
    float lastDTerm() const { return balancePid.lastDTerm(); }

private:
    PID balancePid;
};
