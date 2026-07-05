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
    const float error = target - measurement;

    if (dtSeconds > 0.0f) {
        integral += error * dtSeconds;

        if (hasOutputLimits && ki != 0.0f) {
            const float maxIntegral = maxOutput / fabsf(ki);
            integral = constrain(integral, -maxIntegral, maxIntegral);
        }
    }

    // Use gyro rate for damping; opposes pitch velocity.
    const float derivative = -measurementRate;

    float output = (kp * error) + (ki * integral) + (kd * derivative);

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
    minOutput = minOut;
    maxOutput = maxOut;
    hasOutputLimits = true;
}

void PID::reset() {
    integral = 0.0f;
}
