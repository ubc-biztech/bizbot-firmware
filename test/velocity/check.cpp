// Host regression check: c++ -std=c++11 -Isrc test/velocity/check.cpp -o /tmp/bizbot-velocity-check && /tmp/bizbot-velocity-check
#include "control/VelocityControl.h"
#include <cassert>
#include <cmath>
#include <initializer_list>
int main() {
    // Sustained saturation must not accumulate I in either direction.
    for (float sign : { -1.0f, 1.0f }) {
        float integral = 0;
        for (int n = 0; n < 10000; ++n)
            assert(velocityPI(sign * 100, .1f, .02f, .005f, 3, integral) == sign * 3);
        assert(integral == 0);
        assert(velocityPI(0, .1f, .02f, .005f, 3, integral) == 0);
        // Previously accumulated I can unwind even while saturated.
        integral = sign * 150;
        velocityPI(-sign, .1f, .02f, .005f, 3, integral);
        assert(std::abs(integral) < 150);
        float pitch = sign * 3;
        for (int n = 0; n < 250; ++n) {
            const float next = velocityNeutral(pitch, .015f);
            assert(std::abs(next) <= std::abs(pitch));
            assert(next * sign >= 0);
            pitch = next;
        }
        assert(pitch == 0);
    }
    float integral = 17;
    assert(std::abs(velocityPI(10, .02f, 0, .005f, 3, integral) - .2f) < 1e-6f);
    assert(integral == 0);
    // Elapsed-time integration agrees across regular and delayed cycles.
    float a = 0, b = 0;
    for (int n = 0; n < 10; ++n) velocityPI(1, .02f, .01f, .005f, 3, a);
    velocityPI(1, .02f, .01f, .05f, 3, b);
    assert(std::abs(a - b) < 1e-6f);
}
