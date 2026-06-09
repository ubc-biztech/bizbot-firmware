#pragma once

#include <Arduino.h>
#include "../robot/RobotState.h"

class CommandParser {
public:
    void update(Stream& input, Print& output, RobotState& state);

private:
    static constexpr size_t MAX_COMMAND_LENGTH = 64;

    char buffer[MAX_COMMAND_LENGTH] = {};
    size_t bufferLength = 0;
    bool discardingLine = false;

    void handleCommand(
        const char* command,
        Print& output,
        RobotState& state
    );

    void handleVelocityCommand(
        const char* command,
        Print& output,
        RobotState& state
    );

    void printState(
        Print& output,
        const RobotState& state
    );
};
