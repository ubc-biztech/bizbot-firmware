#include "Imu.h"

#include "../config.h"

#include <Arduino.h>
#include <Wire.h>

bool Imu::begin() {
#if IMU_USE_STUB
    Wire.begin();
    ready = true;
    return true;
#else
    Wire.begin();

    // TODO: Initialize the specific IMU library here.
    // Configure range, sample rate, and filters.
    // Return false if the sensor cannot be detected.

    ready = false;
    return false;
#endif
}

bool Imu::update() {
    if (!ready) {
        return false;
    }

#if IMU_USE_STUB
    pitchDeg = 0.0f;
    pitchRateDegPerSec = 0.0f;
    return true;
#else
    // TODO: Read the IMU and convert mounted axes into robot pitch and rate.
    // pitchDeg = ...;
    // pitchRateDegPerSec = ...;

    return false;
#endif
}

bool Imu::isReady() const {
    return ready;
}

float Imu::getPitchDeg() const {
    return pitchDeg;
}

float Imu::getPitchRateDegPerSec() const {
    return pitchRateDegPerSec;
}
