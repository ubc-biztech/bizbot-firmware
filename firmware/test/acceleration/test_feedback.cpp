#include "../../src/control/AccelerationFeedback.h"
#include <cassert>
#include <initializer_list>
#include <limits>
int main() {
    constexpr float rad = 0.017453292519943295f;
    for (float roll : {-90.f, -40.f, 0.f, 35.f}) {
        for (float pitch : {-30.f, 0.f, 25.f}) {
            float r = roll * rad, p = pitch * rad;
            float gx = -9.80665f * sinf(r) * cosf(p);
            float gy = 9.80665f * sinf(p);
            float gz = 9.80665f * cosf(r) * cosf(p);
            assert(fabsf(horizontalAcceleration(gx, gy, gz, roll, pitch)) < 1e-5f);
            // Add 2 m/s^2 horizontal motion expressed in sensor coordinates.
            float ax = gx + 2 * sinf(p) * sinf(r);
            float ay = gy + 2 * cosf(p);
            float az = gz - 2 * sinf(p) * cosf(r);
            assert(fabsf(horizontalAcceleration(ax, ay, az, roll, pitch)-2) < 1e-5f);
        }
    }
    // Measured stationary sideways mounting: about 1g on X, not Z.
    assert(fabsf(horizontalAcceleration(10.003f, -.804f, 0.f,
                                       -90.14f, -4.65f)) < .03f);
    assert(accelerationCorrection(2, 0, 1, 0) == 0);
    assert(accelerationCorrection(100, .1f, .03f, 0) == -.03f);
    assert(accelerationCorrection(2, .1f, 1, 0) < 0);
    assert(accelerationCorrection(-2, .1f, 1, 0) > 0);
    assert(accelerationCorrection(100, 2, 1, 0) == -1);
    assert(accelerationCorrection(100, 2, 1, 5) == -.5f);
    assert(accelerationCorrection(100, 2, 1, 10) == 0);
    assert(accelerationCorrection(std::numeric_limits<float>::quiet_NaN(), 1, 1, 0) == 0);
}
