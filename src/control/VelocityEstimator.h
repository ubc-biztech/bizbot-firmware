#pragma once
#include <math.h>

// Complementary filter in RPM: integrate gravity-compensated acceleration,
// continuously anchor to wheel speed. tau=0 selects wheel-only feedback.
class VelocityEstimator {
public:
    void reset() { ready = false; }
    float update(float wheelRpm, float acceleration, float dt,
                 float diameterMm, float tau, bool wheelFresh) {
        if (!wheelFresh || !isfinite(wheelRpm)) {
            reset();
            return 0.0f;
        }
        if (!ready || !isfinite(acceleration) || !isfinite(dt) || dt <= 0 || dt > 0.1f ||
            !isfinite(diameterMm) || diameterMm < 50 || !isfinite(tau) || tau <= 0) {
            velocity = wheelRpm;
            ready = true;
            return velocity;
        }
        const float rpmPerMeterPerSecond = 60000.0f / (3.14159265359f * diameterMm);
        const float decay = expf(-dt / tau);
        velocity = wheelRpm + (velocity - wheelRpm) * decay +
            acceleration * rpmPerMeterPerSecond * tau * (1.0f - decay);
        return velocity;
    }
private:
    bool ready = false;
    float velocity = 0;
};
