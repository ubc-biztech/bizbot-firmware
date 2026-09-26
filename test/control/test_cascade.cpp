#include <cassert>
#include <cmath>
#include <initializer_list>
#include "control/BalanceController.h"
#include "control/VelocityController.h"

void near(float actual, float expected) {
    assert(std::fabs(actual - expected) < 1e-5f);
}

VelocityOutput forwardSample(float left, float right) {
    // Supply raw values appropriate to the configured physical wheel signs.
    return calculateVelocityOutput(left / LEFT_WHEEL_VELOCITY_SIGN,
                                   right / RIGHT_WHEEL_VELOCITY_SIGN);
}

int main() {
    // Runtime tuning takes effect without reconstructing/resetting angle control.
    near(calculateVelocityOutput(100 / LEFT_WHEEL_VELOCITY_SIGN,
                                 100 / RIGHT_WHEEL_VELOCITY_SIGN, 0.005f).angleCorrectionDeg, -0.5f);
    near(calculateVelocityOutput(100 / LEFT_WHEEL_VELOCITY_SIGN,
                                 100 / RIGHT_WHEEL_VELOCITY_SIGN, 0.0f).angleCorrectionDeg, 0.0f);
    const auto stopped = forwardSample(0, 0);
    near(stopped.error, 0);
    near(stopped.angleCorrectionDeg, 0);
    const auto forward = forwardSample(100, 60);
    near(forward.left, 100);
    near(forward.right, 60);
    near(forward.forward, 80);
    near(forward.error, -80);
    near(forward.angleCorrectionDeg, -0.8f);
    const auto backward = forwardSample(-100, -60);
    near(backward.angleCorrectionDeg, 0.8f);
    near(forwardSample(100, -100).angleCorrectionDeg, 0);
    near(forwardSample(30000, 30000).angleCorrectionDeg, -MAX_VELOCITY_LEAN_DEG);
    near(forwardSample(-30000, -30000).angleCorrectionDeg, MAX_VELOCITY_LEAN_DEG);
    // P-only has no residual state after sustained saturation.
    for (int i = 0; i < 10000; ++i) forwardSample(30000, 30000);
    near(forwardSample(0, 0).angleCorrectionDeg, 0);

    BalanceController controller(BALANCE_KP, BALANCE_KI, BALANCE_KD);
    // Compare zero outer correction against the original balance equations.
    for (float pitch : {-10.0f, 0.0f, 10.0f}) {
        for (float linear : {-1.0f, 0.0f, 1.0f}) {
            const auto out = controller.update(linear, 0.4f, pitch, 2.0f, CONTROL_DT_SECONDS);
            const float raw = BALANCE_KP * (linear * MAX_LEAN_DEG - pitch) - BALANCE_KD * 2.0f;
            const float limited = std::fmax(-MAX_BALANCE_OUTPUT, std::fmin(MAX_BALANCE_OUTPUT, raw));
            near(out.unclampedSpeed, BALANCE_MOTOR_SIGN * raw);
            near(out.speed, BALANCE_MOTOR_SIGN * limited);
            near(out.steer, 0.4f * MAX_TURN_OUTPUT);
        }
    }
    const auto braking = controller.update(0, 0, 0, 0, CONTROL_DT_SECONDS, forward.angleCorrectionDeg);
    near(braking.speed, BALANCE_MOTOR_SIGN * BALANCE_KP * forward.angleCorrectionDeg);
    const auto saturated = controller.update(1, 2, -30, 0, CONTROL_DT_SECONDS, 5);
    assert(std::fabs(saturated.unclampedSpeed) > MAX_BALANCE_OUTPUT);
    near(std::fabs(saturated.speed), MAX_BALANCE_OUTPUT);
    near(saturated.steer, MAX_TURN_OUTPUT);
    const auto oppositeTurn = controller.update(0, -2, 0, 0, CONTROL_DT_SECONDS, 0);
    near(oppositeTurn.steer, -MAX_TURN_OUTPUT);
    controller.reset();
    near(controller.update(0, 0, 0, 0, CONTROL_DT_SECONDS, 0).speed, 0);
}
