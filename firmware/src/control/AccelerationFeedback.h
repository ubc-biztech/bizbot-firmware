#pragma once
#include <math.h>

// RVC convention observed on this module: body-frame gravity is
// g * [-sin(roll)*cos(pitch), sin(pitch), cos(roll)*cos(pitch)].
// Project onto the horizontal sensor-Y heading. This supports the side mount
// (roll about -90 degrees, gravity on +X) without balance trim or magnetic yaw.
inline float horizontalAcceleration(float ax, float ay, float az,
                                    float rollDeg, float pitchDeg) {
    constexpr float radians = 0.017453292519943295f;
    const float r = rollDeg * radians;
    const float p = pitchDeg * radians;
    return sinf(r) * sinf(p) * ax + cosf(p) * ay -
           cosf(r) * sinf(p) * az;
}

inline float accelerationCorrection(float acceleration, float gain, float limit,
                              float pitch) {
    // Preserve balance recovery authority outside the near-upright region.
    if (!isfinite(acceleration) || !isfinite(gain) || fabsf(pitch) >= 10.0f)
        return 0.0f;
    const float fade = 1.0f - fabsf(pitch) / 10.0f;
    return fmaxf(-limit, fminf(limit, -gain * acceleration)) * fade;
}
