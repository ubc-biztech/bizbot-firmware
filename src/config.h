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

// Set to 1 to compile without a physical IMU (pitch stays at 0; bench use only).
#ifndef IMU_USE_STUB
#define IMU_USE_STUB 0
#endif
