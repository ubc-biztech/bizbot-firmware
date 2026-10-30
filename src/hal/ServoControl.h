#pragma once
#include <Arduino.h>

// Channels 0..2 are reserved for these three servos.
bool handleServoCommand(const char* command, Print& output);
void updateServos();
