#pragma once

class PID {
public:
    PID(float kp, float ki, float kd);

    float update(
        float target,
        float measurement,
        float dtSeconds,
        float measurementRate = 0.0f
    );

    void setTunings(float kp, float ki, float kd);
    void setOutputLimits(float minOutput, float maxOutput);
    void reset();

private:
    float kp;
    float ki;
    float kd;

    float integral = 0.0f;

    float minOutput = -1.0f;
    float maxOutput = 1.0f;
    bool hasOutputLimits = false;
};
