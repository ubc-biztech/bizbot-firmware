#pragma once

class Imu {
public:
    bool begin();
    bool update();
    bool isReady() const;

    float getPitchDeg() const;
    float getPitchRateDegPerSec() const;
    float getForwardAcceleration() const { return forwardAcceleration; }
    float rawAccelX = 0, rawAccelY = 0, rawAccelZ = 0;
    float rawRoll = 0, rawPitch = 0;

private:
    bool ready = false;
    float pitchDeg = 0.0f;
    float pitchRateDegPerSec = 0.0f;
    float forwardAcceleration = 0.0f;
    unsigned long lastReportMs = 0;

    // For deriving pitch rate from RVC (which has no gyro output).
    float prevPitchDeg = 0.0f;
    unsigned long lastSampleUs = 0;
};
