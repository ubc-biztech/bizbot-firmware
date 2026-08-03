#include "PID.h"

#include <Arduino.h>

PID::PID(float kp, float ki, float kd)
    : kp(kp), ki(ki), kd(kd) {}

float PID::update(
    float target,
    float measurement,
    float dtSeconds,
    float measurementRate
) {
    if (dtSeconds <= 0.0f || dtSeconds > 0.1f) {
        return 0.0f;
    }

    const float error = target - measurement;

    integral += error * dtSeconds;

    if (hasOutputLimits && ki != 0.0f) {
        const float maxIntegral = maxOutput / fabsf(ki);
        integral = constrain(integral, -maxIntegral, maxIntegral);
    }

    const float derivative = -measurementRate;

    lastP = kp * error;
    lastI = ki * integral;
    lastD = kd * derivative;

    float output = lastP + lastI + lastD;

    if (hasOutputLimits) {
        output = constrain(output, minOutput, maxOutput);
    }

    return output;
}

void PID::setTunings(float newKp, float newKi, float newKd) {
    kp = newKp;
    ki = newKi;
    kd = newKd;
}

void PID::setOutputLimits(float minOut, float maxOut) {
    if (minOut >= maxOut) {
        return;
    }

    minOutput = minOut;
    maxOutput = maxOut;
    hasOutputLimits = true;

    if (ki != 0.0f) {
        const float maxIntegral = maxOutput / fabsf(ki);
        integral = constrain(integral, -maxIntegral, maxIntegral);
    }
}

void PID::reset() {
    integral = 0.0f;
}
