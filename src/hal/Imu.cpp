#include "Imu.h"

#include "../config.h"

#include <Arduino.h>

#if !IMU_USE_STUB
#include <Adafruit_BNO08x_RVC.h>

namespace {
// Single physical IMU; keep the library object out of the header so nothing
// else that includes Imu.h pulls in the RVC library.
Adafruit_BNO08x_RVC rvc;

// The BNO085 UART-RVC stream is on UART1 (UART2 belongs to the hoverboard).
HardwareSerial ImuSerial(1);

// Select the mounted pitch axis from the RVC Euler angles (degrees in).
float selectPitchDeg(float roll, float pitch, float yaw) {
    switch (IMU_PITCH_SOURCE) {
        case IMU_PITCH_FROM_ROLL: return roll;
        case IMU_PITCH_FROM_YAW:  return yaw;
        default:                  return pitch;
    }
}
}  // namespace
#endif

bool Imu::begin() {
#if IMU_USE_STUB
    ready = true;
    return true;
#else
    // RVC is a one-way stream (sensor TX -> ESP32 RX only); no TX pin needed.
    ImuSerial.begin(IMU_UART_BAUD, SERIAL_8N1, IMU_UART_RX_PIN, -1);

#if IMU_UART_RAW_DUMP
    // Diagnostic: sample raw bytes for 3 s so we can see if anything arrives.
    Serial.print("IMU_RAW: sampling GPIO ");
    Serial.print(IMU_UART_RX_PIN);
    Serial.println(" @115200 for 3s...");
    uint32_t rawCount = 0;
    const unsigned long rawStart = millis();
    while (millis() - rawStart < 3000) {
        while (ImuSerial.available()) {
            const uint8_t b = ImuSerial.read();
            if (rawCount < 48) {
                if (b < 0x10) Serial.print('0');
                Serial.print(b, HEX);
                Serial.print(' ');
            }
            rawCount++;
        }
        delay(1);
    }
    Serial.print("\nIMU_RAW: ");
    Serial.print(rawCount);
    Serial.println(" bytes in 3s");
#endif

    if (!rvc.begin(&ImuSerial)) {
        ready = false;
        return false;
    }

    // RVC has no handshake, so presence is confirmed by receiving a real packet.
    BNO08x_RVC_Data packet;
    const unsigned long start = millis();
    while (millis() - start < IMU_UART_DETECT_MS) {
        if (rvc.read(&packet)) {
            pitchDeg = IMU_PITCH_SIGN *
                       selectPitchDeg(packet.roll, packet.pitch, packet.yaw);
            prevPitchDeg = pitchDeg;
            pitchRateDegPerSec = 0.0f;
            lastSampleUs = micros();
            lastReportMs = millis();
            ready = true;
            return true;
        }
    }

    ready = false;
    return false;
#endif
}

bool Imu::update() {
    if (!ready) {
        return false;
    }

#if IMU_USE_STUB
    pitchDeg = 0.0f;
    pitchRateDegPerSec = 0.0f;
    return true;
#else
    // Drain all queued RVC packets (non-blocking); act on the freshest one.
    BNO08x_RVC_Data packet;
    bool gotReport = false;
    while (rvc.read(&packet)) {
        gotReport = true;
    }

    if (gotReport) {
        pitchDeg = IMU_PITCH_SIGN *
                   selectPitchDeg(packet.roll, packet.pitch, packet.yaw);

        // UART-RVC has no gyro; derive pitch rate by differentiating pitch and
        // low-pass filtering. Rate inherits the sign already baked into pitchDeg.
        const unsigned long nowUs = micros();
        const float dt = (nowUs - lastSampleUs) * 1e-6f;
        if (lastSampleUs != 0 && dt > 0.0f) {
            const float rawRate = (pitchDeg - prevPitchDeg) / dt;
            pitchRateDegPerSec +=
                IMU_RATE_LPF_ALPHA * (rawRate - pitchRateDegPerSec);
        }
        lastSampleUs = nowUs;
        prevPitchDeg = pitchDeg;

        lastReportMs = millis();

#if IMU_DEBUG_EULER
        static uint32_t lastDebugMs = 0;
        const uint32_t nowMs = millis();
        if (nowMs - lastDebugMs >= 200) {
            lastDebugMs = nowMs;
            Serial.print("IMU rpy(deg)=");
            Serial.print(packet.roll, 1);
            Serial.print(',');
            Serial.print(packet.pitch, 1);
            Serial.print(',');
            Serial.print(packet.yaw, 1);
            Serial.print(" pitch=");
            Serial.print(pitchDeg, 1);
            Serial.print(" rate(dps)=");
            Serial.println(pitchRateDegPerSec, 1);
        }
#endif
    }

    // Stale data (sensor unresponsive / wiring fault) is a control fault.
    return (millis() - lastReportMs) <= IMU_STALE_TIMEOUT_MS;
#endif
}

bool Imu::isReady() const {
    return ready;
}

float Imu::getPitchDeg() const {
    return pitchDeg;
}

float Imu::getPitchRateDegPerSec() const {
    return pitchRateDegPerSec;
}
