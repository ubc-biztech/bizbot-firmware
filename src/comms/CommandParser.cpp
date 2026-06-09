#include "CommandParser.h"

namespace {
constexpr float MIN_COMMAND_VALUE = -1.0f;
constexpr float MAX_COMMAND_VALUE = 1.0f;
}

// While input is available:
    // 1. Read one character.
    // 2. Add it to buffer.
    // 3. On '\n', call handleCommand().
    // 4. Reset the buffer.
    // 5. Reject commands that are too long.
    
void CommandParser::update(
    Stream& input,
    Print& output,
    RobotState& state
) {
    while (input.available() > 0) {
        const char character = static_cast<char>(input.read());

        if (character == '\r') {
            continue;
        }

        if (character == '\n') {
            if (discardingLine) {
                discardingLine = false;
                bufferLength = 0;
                output.println("ERR COMMAND_TOO_LONG");
                continue;
            }

            buffer[bufferLength] = '\0';

            if (bufferLength > 0) {
                handleCommand(buffer, output, state);
            }

            bufferLength = 0;
            continue;
        }

        if (discardingLine) {
            continue;
        }

        // Keep one byte available for the terminating '\0'.
        if (bufferLength >= MAX_COMMAND_LENGTH - 1) {
            discardingLine = true;
            bufferLength = 0;
            continue;
        }

        buffer[bufferLength] = character;
        bufferLength++;
    }
}

void CommandParser::handleCommand(
    const char* command,
    Print& output,
    RobotState& state
) {
    if (strcmp(command, "ENABLE") == 0) {
        state.enabled = true;
        state.lastCommandMs = millis();
        output.println("OK ENABLED");
        return;
    }

    if (strcmp(command, "DISABLE") == 0) {
        state.enabled = false;
        state.targetLinear = 0.0f;
        state.targetAngular = 0.0f;
        output.println("OK DISABLED");
        return;
    }

    if (strcmp(command, "STOP") == 0) {
        state.targetLinear = 0.0f;
        state.targetAngular = 0.0f;
        state.lastCommandMs = millis();
        output.println("OK STOP");
        return;
    }

    if (strncmp(command, "CMD_VEL ", 8) == 0) {
        handleVelocityCommand(command, output, state);
        return;
    }

    if (strcmp(command, "GET_STATE") == 0) {
        printState(output, state);
        return;
    }

    output.println("ERR UNKNOWN_COMMAND");
}

void CommandParser::handleVelocityCommand(
    const char* command,
    Print& output,
    RobotState& state
) {
    if (!state.enabled) {
        output.println("ERR DISABLED");
        return;
    }

    float linear = 0.0f;
    float angular = 0.0f;
    char extra = '\0';

    const int parsed = sscanf(
        command,
        "CMD_VEL %f %f %c",
        &linear,
        &angular,
        &extra
    );

    if (parsed != 2) {
        output.println("ERR INVALID_ARGUMENTS");
        return;
    }

    const bool linearOutOfRange =
        linear < MIN_COMMAND_VALUE || linear > MAX_COMMAND_VALUE;
    const bool angularOutOfRange =
        angular < MIN_COMMAND_VALUE || angular > MAX_COMMAND_VALUE;

    if (linearOutOfRange || angularOutOfRange) {
        output.println("ERR OUT_OF_RANGE");
        return;
    }

    state.targetLinear = linear;
    state.targetAngular = angular;
    state.lastCommandMs = millis();
    output.println("OK CMD_VEL");
}

void CommandParser::printState(
    Print& output,
    const RobotState& state
) {
    output.print("STATE enabled=");
    output.print(state.enabled ? 1 : 0);
    output.print(" linear=");
    output.print(state.targetLinear, 3);
    output.print(" angular=");
    output.print(state.targetAngular, 3);
    output.print(" pitch=");
    output.print(state.pitchDeg, 2);
    output.print(" battery=");
    output.println(state.batteryVoltage, 2);
}
