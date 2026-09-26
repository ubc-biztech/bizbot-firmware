#pragma once

#include "../config.h"

struct VelocityOutput {
    float left;
    float right;
    float forward;
    float error;
    float angleCorrectionDeg;
};

// Stateless P controller. Scheduling and sample-and-hold live in main.cpp.
// No integral is accumulated while Ki=0; adding I later requires anti-windup.
inline VelocityOutput calculateVelocityOutput(float rawLeft, float rawRight) {
    VelocityOutput result{};
    result.left = LEFT_WHEEL_VELOCITY_SIGN * rawLeft;
    result.right = RIGHT_WHEEL_VELOCITY_SIGN * rawRight;
    result.forward = 0.5f * (result.left + result.right);
    result.error = TARGET_FORWARD_VELOCITY - result.forward;
    // Forward velocity -> negative error -> backward lean (negative pitch).
    // Backward velocity -> positive error -> forward lean (positive pitch).
    const float correction = VELOCITY_ANGLE_SIGN * VELOCITY_KP * result.error;
    result.angleCorrectionDeg = correction > MAX_VELOCITY_LEAN_DEG
        ? MAX_VELOCITY_LEAN_DEG
        : (correction < -MAX_VELOCITY_LEAN_DEG ? -MAX_VELOCITY_LEAN_DEG : correction);
    return result;
}
