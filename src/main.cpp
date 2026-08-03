/*
 * BizBot ESP32 firmware — balance control over hoverboard UART.
 *
 * Wiring (Right Sideboard connector = USART3, 5V tolerant):
 *   Mainboard PB10 (TX,  blue)  -> ESP32 RX  (HOVER_RX_PIN)
 *   Mainboard PB11 (RX,  green) -> ESP32 TX  (HOVER_TX_PIN)
 *   Mainboard GND (black)       -> ESP32 GND
 *   Mainboard 15V (red)         -> DO NOT CONNECT
 */

#include <Arduino.h>
#include <WiFi.h>

#include "comms/CommandParser.h"
#include "config.h"
#include "control/BalanceController.h"
#include "hal/Imu.h"
#include "robot/RobotState.h"

HardwareSerial& HoverSerial = Serial2;
HardwareSerial& DebugSerial = Serial;

RobotState robotState;
CommandParser usbCommandParser;
CommandParser wifiCommandParser;
Imu imu;
BalanceController balanceController(
    BALANCE_KP,
    BALANCE_KI,
    BALANCE_KD
);
WiFiServer wifiControlServer(WIFI_CONTROL_PORT);
WiFiClient wifiControlClient;

typedef struct {
    uint16_t start;
    int16_t steer;
    int16_t speed;
    uint16_t checksum;
} SerialCommand;

typedef struct {
    uint16_t start;
    int16_t cmd1;
    int16_t cmd2;
    int16_t speedR_meas;
    int16_t speedL_meas;
    int16_t batVoltage;
    int16_t boardTemp;
    uint16_t cmdLed;
    uint16_t checksum;
} SerialFeedback;

SerialCommand Command;
SerialFeedback Feedback;
SerialFeedback NewFeedback;

uint32_t lastControlUs = 0;
bool latchedImuFault = false;
bool wasEnabled = false;
bool wifiControlReady = false;
bool wifiControlWasConnected = false;

void onEnabled() {
    balanceController.reset();
    latchedImuFault = false;
    robotState.targetLinear = 0.0f;
    robotState.targetAngular = 0.0f;
}

void Send(int16_t steer, int16_t speed) {
    Command.start = START_FRAME;
    Command.steer = steer;
    Command.speed = speed;
    Command.checksum =
        static_cast<uint16_t>(Command.start ^ Command.steer ^ Command.speed);
    HoverSerial.write(reinterpret_cast<uint8_t*>(&Command), sizeof(Command));
}

void disableRobot() {
    robotState.enabled = false;
    robotState.targetLinear = 0.0f;
    robotState.targetAngular = 0.0f;
    wasEnabled = false;

    balanceController.reset();
    Send(0, 0);
}

void Receive() {
    static uint8_t idx = 0;
    static uint8_t prev = 0;
    static uint8_t* p = reinterpret_cast<uint8_t*>(&NewFeedback);

    while (HoverSerial.available()) {
        const uint8_t in = HoverSerial.read();
        const uint16_t frame = (static_cast<uint16_t>(in) << 8) | prev;

        if (frame == START_FRAME) {
            p = reinterpret_cast<uint8_t*>(&NewFeedback);
            *p++ = prev;
            *p++ = in;
            idx = 2;
        } else if (idx >= 2 && idx < sizeof(SerialFeedback)) {
            *p++ = in;
            idx++;
        }

        if (idx == sizeof(SerialFeedback)) {
            const uint16_t checksum = static_cast<uint16_t>(
                NewFeedback.start ^ NewFeedback.cmd1 ^ NewFeedback.cmd2 ^
                NewFeedback.speedR_meas ^ NewFeedback.speedL_meas ^
                NewFeedback.batVoltage ^ NewFeedback.boardTemp ^
                NewFeedback.cmdLed);

            if (
                NewFeedback.start == START_FRAME &&
                checksum == NewFeedback.checksum
            ) {
                Feedback = NewFeedback;

                robotState.batteryVoltage =
                    Feedback.batVoltage / 100.0f;
                robotState.leftWheelSpeed =
                    static_cast<float>(Feedback.speedL_meas);
                robotState.rightWheelSpeed =
                    static_cast<float>(Feedback.speedR_meas);
                robotState.lastMotorFeedbackMs = millis();
            }

            idx = 0;
        }

        prev = in;
    }
}

int16_t normalizedToHoverboard(float value) {
    value = constrain(value, -1.0f, 1.0f);
    return static_cast<int16_t>(value * MAX_HOVERBOARD_COMMAND);
}

bool commandTimedOut() {
    return millis() - robotState.lastCommandMs > COMMAND_TIMEOUT_MS;
}

bool tiltLimitExceeded() {
    return fabsf(robotState.pitchDeg) > MAX_TILT_DEG;
}

void stopRobot(const char* reason) {
    disableRobot();

    DebugSerial.print("FAULT ");
    DebugSerial.println(reason);
}

void startWifiControl() {
    WiFi.mode(WIFI_AP);

    if (!WiFi.softAP(WIFI_AP_SSID, WIFI_AP_PASSWORD)) {
        DebugSerial.println("FAULT WIFI_AP");
        return;
    }

    wifiControlServer.begin();
    wifiControlServer.setNoDelay(true);
    wifiControlReady = true;

    DebugSerial.print("WiFi AP: ");
    DebugSerial.println(WIFI_AP_SSID);
    DebugSerial.print("WiFi control: ");
    DebugSerial.print(WiFi.softAPIP());
    DebugSerial.print(":");
    DebugSerial.println(WIFI_CONTROL_PORT);
}

void discardUsbCommands() {
    while (DebugSerial.available() > 0) {
        DebugSerial.read();
    }
}

void updateCommandInput() {
    bool wifiConnected =
        wifiControlClient && wifiControlClient.connected();

    if (!wifiConnected && wifiControlWasConnected) {
        wifiControlClient.stop();
        stopRobot("WIFI_DISCONNECT");
        DebugSerial.println("WiFi controller disconnected; USB control restored");
    }

    if (!wifiConnected && wifiControlReady) {
        WiFiClient candidate = wifiControlServer.available();

        if (candidate) {
            disableRobot();
            wifiControlClient = candidate;
            wifiControlClient.setNoDelay(true);
            wifiConnected = true;

            DebugSerial.print("WiFi controller connected: ");
            DebugSerial.println(wifiControlClient.remoteIP());
        }
    }

    wifiControlWasConnected = wifiConnected;

    if (wifiConnected) {
        // Wi-Fi owns control while connected. Do not queue stale USB commands.
        discardUsbCommands();
        wifiCommandParser.update(
            wifiControlClient,
            wifiControlClient,
            robotState
        );
        return;
    }

    usbCommandParser.update(DebugSerial, DebugSerial, robotState);
}

void runControlLoop() {
#if MOTOR_TEST
    // UART link test: constant slow spin on both wheels, everything else
    // bypassed (including the IMU, so it works with no sensor attached).
    Send(0, MOTOR_TEST_SPEED);
    return;
#endif

    if (!imu.isReady()) {
        if (robotState.enabled && !latchedImuFault) {
            latchedImuFault = true;
            stopRobot("IMU_INIT");
        }
        Send(0, 0);
        return;
    }

    if (!imu.update()) {
        if (!latchedImuFault) {
            latchedImuFault = true;
            stopRobot("IMU");
        }
        Send(0, 0);
        return;
    }

    latchedImuFault = false;

    robotState.pitchDeg = imu.getPitchDeg();
    robotState.pitchRateDegPerSec = imu.getPitchRateDegPerSec();
    robotState.lastImuUpdateMs = millis();

    if (!robotState.enabled) {
        balanceController.reset();
        Send(0, 0);
        return;
    }

    if (commandTimedOut()) {
        stopRobot("COMMAND_TIMEOUT");
        return;
    }

    if (tiltLimitExceeded()) {
        stopRobot("TILT_LIMIT");
        return;
    }

    const BalanceOutput output = balanceController.update(
        robotState.targetLinear,
        robotState.targetAngular,
        robotState.pitchDeg,
        robotState.pitchRateDegPerSec,
        CONTROL_DT_SECONDS
    );

    const int16_t speed = normalizedToHoverboard(output.speed);
    const int16_t steer = normalizedToHoverboard(output.steer);

    Send(steer, speed);
}

void setup() {
    DebugSerial.begin(115200);

    HoverSerial.begin(
        HOVER_SERIAL_BAUD,
        SERIAL_8N1,
        HOVER_RX_PIN,
        HOVER_TX_PIN
    );

    Send(0, 0);
    startWifiControl();

    if (!imu.begin()) {
        robotState.enabled = false;
        DebugSerial.println("FAULT IMU_INIT");
    } else {
        DebugSerial.println("OK IMU_INIT");
    }

    robotState.lastCommandMs = millis();
    DebugSerial.println("BizBot firmware ready");
}

void loop() {
    updateCommandInput();

    if (robotState.enabled && !wasEnabled) {
        onEnabled();
    }
    wasEnabled = robotState.enabled;

    Receive();

    const uint32_t now = micros();
    if (now - lastControlUs >= CONTROL_PERIOD_US) {
        lastControlUs += CONTROL_PERIOD_US;
        runControlLoop();
    }
}
