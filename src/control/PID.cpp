#include "PID.h"

#include <Arduino.h>
#include <math.h>

PID::PID(float kp, float ki, float kd)
    : kp(kp), kpBack(kp), ki(ki), kd(kd) {}

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

    // Smooth, bounded P boost around the target angle.
    const float directionalKp = error < 0.0f ? kp : kpBack;
    lastP = directionalKp * (error + proportionalBoost * tanhf(error / proportionalWidth));
    lastI = ki * integral;
    // D stays on measurement rate; angle error only schedules its gain.
    // Width is the angle error where half the extra D gain is applied.
    const float errorSquared = error * error;
    const float dScale = 1.0f + derivativeBoost * errorSquared /
        (errorSquared + derivativeWidth * derivativeWidth);
    lastD = kd * dScale * derivative;

    float output = lastP + lastI + lastD;

    if (hasOutputLimits) {
        output = constrain(output, minOutput, maxOutput);
    }

    return output;
}

void PID::setTunings(float newKp, float newKi, float newKd) {
    kp = newKp;
    kpBack = newKp;
    ki = newKi;
    kd = newKd;
}

void PID::setProportionalBoost(float errorBoost, float errorWidth) {
    if (!isfinite(errorBoost) || !isfinite(errorWidth) ||
        errorBoost < 0.0f || errorWidth <= 0.0f) {
        return;
    }
    proportionalBoost = errorBoost;
    proportionalWidth = errorWidth;
}

void PID::setDirectionalP(float front, float back) {
    if (!isfinite(front) || !isfinite(back) || front < 0.0f || back < 0.0f) return;
    kp = front;
    kpBack = back;
}

void PID::setDerivativeBoost(float boost, float width) {
    if (!isfinite(boost) || !isfinite(width) || boost < 0.0f || width <= 0.0f) return;
    derivativeBoost = boost;
    derivativeWidth = width;
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
