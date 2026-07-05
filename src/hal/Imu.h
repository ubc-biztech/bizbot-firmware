#pragma once

class Imu {
public:
    bool begin();
    bool update();
    bool isReady() const;

    float getPitchDeg() const;
    float getPitchRateDegPerSec() const;

private:
    bool ready = false;
    float pitchDeg = 0.0f;
    float pitchRateDegPerSec = 0.0f;
};
