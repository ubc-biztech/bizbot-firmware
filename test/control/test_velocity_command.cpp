#include <cassert>
#include "comms/CommandParser.h"
#include "control/VelocityController.h"

unsigned long millis() { return 1234; }
class Input : public Stream {
public:
    std::string data;
    size_t position = 0;
    int available() override { return static_cast<int>(data.size() - position); }
    int read() override { return data[position++]; }
};

int main() {
    CommandParser parser;
    RobotState state;
    state.enabled = true;
    state.balanceKp = BALANCE_KP;
    state.pitchTrimDeg = PITCH_TRIM_DEG;
    state.velocityKp = VELOCITY_KP;
    auto command = [&](const std::string& text) {
        Input input;
        input.data = text + "\n";
        Print output;
        parser.update(input, output, state);
        return output.text;
    };
    assert(command("SET_VEL_KP 0.005") == "OK SET_VEL_KP 0.00500\n");
    assert(state.velocityKp == 0.005f);
    assert(state.lastCommandMs == 1234);
    assert(state.enabled && state.balanceKp == BALANCE_KP && state.pitchTrimDeg == PITCH_TRIM_DEG);
    assert(!state.resetBalanceRequested);
    assert(calculateVelocityOutput(100, 100, state.velocityKp).angleCorrectionDeg == -0.5f);
    for (const char* invalid : {"SET_VEL_KP", "SET_VEL_KP nope", "SET_VEL_KP 0.1 extra",
                                "SET_VEL_KP nan", "SET_VEL_KP inf", "SET_VEL_KP -0.1",
                                "SET_VEL_KP 1.001", "SET_VEL_KP 1e99"}) {
        assert(command(invalid).find("ERR ") == 0);
        assert(state.velocityKp == 0.005f);
    }
    assert(command("GET_STATE").find("vel_kp=0.00500") != std::string::npos);
    command("DISABLE");
    command("ENABLE");
    assert(state.velocityKp == 0.005f);
    assert(command("SET_VEL_KP 0") == "OK SET_VEL_KP 0.00000\n");
    assert(calculateVelocityOutput(100, 100, state.velocityKp).angleCorrectionDeg == 0);
    assert(command("SET_VEL_KP 1") == "OK SET_VEL_KP 1.00000\n");
}
