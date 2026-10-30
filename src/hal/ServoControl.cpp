#include "ServoControl.h"
#include "../config.h"
#include <math.h>

namespace {
const int pins[] = {SERVO_1_PIN, SERVO_2_PIN, SERVO_3_PIN};
bool attached[3] = {};
unsigned long lastContact = 0;
void release(int i) {
    if (!attached[i]) return;
    ledcWrite(i, 0);
    ledcDetachPin(pins[i]);
    pinMode(pins[i], OUTPUT);
    digitalWrite(pins[i], LOW);
    attached[i] = false;
}
}

bool handleServoCommand(const char* command, Print& output) {
    if (strcmp(command, "SERVO_PING") == 0) {
        lastContact = millis();
        output.println("OK SERVO_PING");
        return true;
    }
    if (strcmp(command, "SERVO_OFF") == 0) {
        for (int i = 0; i < 3; ++i) release(i);
        output.println("OK SERVO_OFF");
        return true;
    }
    if (strncmp(command, "SERVO_OFF ", 10) == 0) {
        int id; char extra;
        if (sscanf(command, "SERVO_OFF %d %c", &id, &extra) != 1 || id < 1 || id > 3) {
            output.println("ERR SERVO_ARGUMENTS");
        } else {
            release(id - 1);
            output.printf("OK SERVO_OFF %d\n", id);
        }
        return true;
    }
    if (strncmp(command, "SERVO ", 6) != 0) return false;
    int id; float angle; char extra;
    if (sscanf(command, "SERVO %d %f %c", &id, &angle, &extra) != 2 ||
        id < 1 || id > 3 || !isfinite(angle) || angle < 0 || angle > 180) {
        output.println("ERR SERVO_ARGUMENTS");
        return true;
    }
    const int i = id - 1;
    if (!attached[i]) {
        if (ledcSetup(i, 50, 16) == 0) {
            output.println("ERR SERVO_PWM");
            return true;
        }
        ledcWrite(i, 0);
        ledcAttachPin(pins[i], i);
        attached[i] = true;
    }
    const float pulse = SERVO_MIN_PULSE_US + angle / 180.0f *
        (SERVO_MAX_PULSE_US - SERVO_MIN_PULSE_US);
    ledcWrite(i, static_cast<uint32_t>(lroundf(pulse * 65536.0f / 20000.0f)));
    lastContact = millis();
    output.printf("OK SERVO %d %.1f\n", id, angle);
    return true;
}

void updateServos() {
    if (millis() - lastContact > 2000) {
        for (int i = 0; i < 3; ++i) release(i);
    }
}
