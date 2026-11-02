#include <Arduino.h>
#include <ESP32Servo.h>

// Standalone ESP32 servo test using ESP32Servo. No BizBot code.
constexpr int SERVO_PINS[] = {18, 21, 23};
constexpr int MIN_PULSE_US = 1000;
constexpr int MAX_PULSE_US = 2000;
constexpr int FRAME_US = 20000; // 50 Hz

Servo s1;
Servo s2;
Servo s3;

void setup() {
    Serial.begin(115200);
    s1.attach(SERVO_PINS[0]);
    s2.attach(SERVO_PINS[1]);
    s3.attach(SERVO_PINS[2]);
}

void loop() {

    s1.write(90);
    s2.write(90);
    s3.write(90);


    delay(1000);
    s1.write(0);
    s2.write(0);
    s3.write(0);

    delay(1000);
}
