#pragma once

#include <stdint.h>

// Hoverboard UART (EFeru hoverboard-firmware-hack-FOC, VARIANT_USART on USART3)
constexpr uint32_t HOVER_SERIAL_BAUD = 115200;
constexpr uint16_t START_FRAME = 0xABCD;
constexpr int HOVER_RX_PIN = 25;
constexpr int HOVER_TX_PIN = 26;

// Local Wi-Fi control. Change the password before using the robot in public.
constexpr char WIFI_AP_SSID[] = "BizBot-Control";
constexpr char WIFI_AP_PASSWORD[] = "bizbot-control";
constexpr uint16_t WIFI_CONTROL_PORT = 3333;

// Max hoverboard command magnitude; confirm with restrained hardware tests.
constexpr int16_t MAX_HOVERBOARD_COMMAND = 50;

// Control loop timing
constexpr uint32_t CONTROL_PERIOD_US = 5000;
constexpr float CONTROL_DT_SECONDS = 0.005f;

// Balance PID placeholders — tune on the physical robot with restrained output.
constexpr float BALANCE_KP = 0.04f;
constexpr float BALANCE_KI = 0.0f;
constexpr float BALANCE_KD = 0.001f;

// Balance controller output limits (normalized [-1, 1])
constexpr float MAX_LEAN_DEG = 5.0f;
constexpr float MAX_BALANCE_OUTPUT = 0.20f;
constexpr float MAX_TURN_OUTPUT = 0.15f;

// Laptop keepalive should be faster than this (keyboard_controls.py uses 200 ms).
constexpr unsigned long COMMAND_TIMEOUT_MS = 500;
constexpr float MAX_TILT_DEG = 35.0f;

// ---------------------------------------------------------------------------
// IMU: GY-BM008X = Bosch/CEVA BNO085 (BNO08x family), UART-RVC mode.
// The BNO085 does sensor fusion onboard. In UART-RVC mode it streams a fixed
// 19-byte packet (pitch/roll/yaw in degrees + accel) at ~100 Hz over one wire.
//
// UART is used because the BNO08x's I2C violates the I2C spec and wedges the
// ESP32 I2C peripheral. UART-RVC is the simplest reliable path on ESP32.
//
// Wiring (UART-RVC mode — note PS pins differ from I2C/SPI):
//   Module VCC     -> ESP32 3V3     Module GND -> ESP32 GND
//   Module SDA     -> IMU_UART_RX_PIN  (RVC data-out -> ESP32 RX; one-way stream)
//   Module PS1     -> GND           (selects UART-RVC)
//   Module PS0     -> 3V3           (selects UART-RVC)
// The hoverboard already owns UART2 (Serial2); the IMU uses UART1 (Serial1).
// ---------------------------------------------------------------------------
constexpr int IMU_UART_RX_PIN = 32;        // sensor SDA (data-out) -> ESP32 RX (UART1)
constexpr uint32_t IMU_UART_BAUD = 115200; // fixed by UART-RVC mode

// begin() waits this long for the first valid RVC packet before declaring the
// IMU absent (RVC is one-way, so this is our only presence check).
constexpr unsigned long IMU_UART_DETECT_MS = 500;

// No RVC packet within this window => IMU fault (latched stop in main.cpp).
constexpr unsigned long IMU_STALE_TIMEOUT_MS = 100;

// --- Mounting orientation --------------------------------------------------
// RVC reports roll/pitch/yaw directly (degrees); pick the axis that tracks the
// robot leaning forward/back over its wheel axle.
//
// Bench calibration: build with IMU_DEBUG_EULER=1, tip the robot FORWARD, and
// watch the serial dump. Set IMU_PITCH_SOURCE to the Euler angle that changes,
// and IMU_PITCH_SIGN so forward lean gives a POSITIVE pitchDeg.
#define IMU_PITCH_FROM_ROLL  0
#define IMU_PITCH_FROM_PITCH 1
#define IMU_PITCH_FROM_YAW   2
constexpr int   IMU_PITCH_SOURCE = IMU_PITCH_FROM_PITCH;
constexpr float IMU_PITCH_SIGN   = 1.0f;

// UART-RVC gives no gyro, so pitch rate is derived by differentiating pitch and
// low-pass filtering. Alpha in (0,1]: higher = more responsive/noisier, lower =
// smoother/laggier. Rate inherits IMU_PITCH_SIGN via pitchDeg, so no rate sign.
constexpr float IMU_RATE_LPF_ALPHA = 0.3f;

// Set to 1 to periodically print roll/pitch/yaw + derived rate for the
// orientation calibration above. Leave at 0 for normal operation.
#ifndef IMU_DEBUG_EULER
#define IMU_DEBUG_EULER 0
#endif

// Set to 1 to sample raw bytes off the IMU UART pin at boot and print them as
// hex (a diagnostic to see whether the sensor is transmitting at all). The RVC
// framing byte is 0xAA — a valid stream shows repeating "AA AA ...". Leave 0.
#ifndef IMU_UART_RAW_DUMP
#define IMU_UART_RAW_DUMP 0
#endif

// Set to 1 to compile without a physical IMU (pitch stays at 0; bench use only).
#ifndef IMU_USE_STUB
#define IMU_USE_STUB 0
#endif

// Set to 1 to bypass ALL control logic and safety gates and spin both wheels
// slowly at a constant command — a hoverboard UART link test. The wheels turn
// as soon as the board powers up, with no ENABLE, no keepalive, and no tilt
// cutoff. WHEELS OFF THE GROUND. Leave 0 for normal operation.
#ifndef MOTOR_TEST
#define MOTOR_TEST 0
#endif

// Constant speed command sent while MOTOR_TEST=1 (EFeru range is -1000..1000;
// wheels typically start turning around 25-50).
constexpr int16_t MOTOR_TEST_SPEED = 100;
