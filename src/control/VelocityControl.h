#pragma once
#include <math.h>

inline float velocityNeutral(float pitch, float step) {
    return pitch > step ? pitch - step : (pitch < -step ? pitch + step : 0.0f);
}

// Integrate only when the combined P+I output is not being pushed farther
// into saturation. Keep the integral bounded, and allow it to unwind.
inline float velocityPI(float error, float kp, float ki, float dt,
                        float limit, float& integral) {
    if (ki <= 0.0f) {
        integral = 0.0f;
    } else {
        const float bound = limit / ki;
        integral = fmaxf(-bound, fminf(bound, integral));
        const float candidate = fmaxf(-bound, fminf(bound, integral + error * dt));
        const float proposed = kp * error + ki * candidate;
        if ((proposed <= limit || error < 0.0f) &&
            (proposed >= -limit || error > 0.0f)) {
            integral = candidate;
        }
    }
    return fmaxf(-limit, fminf(limit, kp * error + ki * integral));
}
