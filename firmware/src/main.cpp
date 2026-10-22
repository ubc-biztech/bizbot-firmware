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
#include <string.h>
#include <WiFi.h>
#include <lwip/sockets.h>

#include "comms/CommandParser.h"
#include "hal/ServoControl.h"
#include "config.h"
#include "control/BalanceController.h"
#include "control/AccelerationFeedback.h"
#include "control/VelocityControl.h"
#include "control/VelocityEstimator.h"
#include "hal/Imu.h"
#include "hal/OffsetStorage.h"
#include "robot/RobotState.h"
#include "control/bizbot_policy.h"

HardwareSerial& HoverSerial = Serial2;
HardwareSerial& DebugSerial = Serial;

RobotState robotState;
CommandParser usbCommandParser;
CommandParser wifiCommandParser;
Imu imu;
VelocityEstimator velocityEstimator;
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

// RL policy (O 120M) drives instead of the PID. The PID still runs and its command is logged.
// Policy runs every 2nd 5 ms tick (its 10 ms period); DZ_COMP 20 is added to every nonzero
// command, as in training. One line per policy run, never blocking:
//   RL <ms> <pitch_deg> <accel_fwd> <wheel_rpm> <battery_v> <pid_cmd> <rl_cmd> <us>
constexpr float RL_DRIVE_SIGN = -BALANCE_MOTOR_SIGN;
constexpr int16_t RL_DZ_COMP = 20;
BizbotPolicyState rlState;
int rlTickPhase = 0;
int16_t rlAppliedCmd = 0;
int16_t rlCmdHeld = 0;

void onEnabled() {
    bizbot_policy_reset(&rlState);
    rlTickPhase = 0;
    rlAppliedCmd = 0;
    rlCmdHeld = 0;
    balanceController.reset();
    robotState.velIntegral = 0.0f;
    robotState.targetPitchDeg = 0.0f;
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
    robotState.velIntegral = 0.0f;
    robotState.targetPitchDeg = 0.0f;
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
                robotState.wheelVelocity = 0.5f * (
                    robotState.leftWheelSpeed +
                    WHEEL_RIGHT_SIGN * robotState.rightWheelSpeed);
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
    velocityEstimator.reset();
    disableRobot();

    DebugSerial.print("FAULT ");
    DebugSerial.println(reason);
}

// Outer loop: wheel RPM -> target pitch. Losing feedback returns the target
// to neutral without disabling the inner balance loop.
float velocityLoop(float dtSeconds) {
    const bool active = robotState.velKp > 0.0f || robotState.velKi > 0.0f;
    if (!active) {
        robotState.velIntegral = 0.0f;
        return robotState.targetLinear * MAX_LEAN_DEG;
    }

    const bool fresh = robotState.lastMotorFeedbackMs != 0 &&
        millis() - robotState.lastMotorFeedbackMs <= WHEEL_FEEDBACK_STALE_MS;
    if (!fresh) {
        robotState.velIntegral = 0.0f;
        return velocityNeutral(robotState.targetPitchDeg,
            WHEEL_STALE_RETURN_DEG_PER_SEC * dtSeconds);
    }

    const float vTarget = robotState.targetLinear * MAX_WHEEL_RPM;
    const float vErr = vTarget - robotState.fusedVelocity;

    return robotState.velSign * velocityPI(vErr, robotState.velKp,
        robotState.velKi, dtSeconds, MAX_VEL_LEAN_DEG, robotState.velIntegral);
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

void runControlLoop(float dtSeconds) {
    if (robotState.resetBalanceRequested) {
        balanceController.reset();
        robotState.velIntegral = 0.0f;
        robotState.targetPitchDeg = 0.0f;
        robotState.resetBalanceRequested = false;
        velocityEstimator.reset();
    }
#if IMU_USE_STUB
    // The stub reports a permanently upright robot and must never drive motors.
    if (robotState.enabled) {
        stopRobot("IMU_STUB");
    } else {
        Send(0, 0);
    }
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

    robotState.pitchDeg = imu.getPitchDeg() - robotState.pitchTrimDeg;
    robotState.pitchRateDegPerSec = imu.getPitchRateDegPerSec();
    robotState.forwardAccel = imu.getForwardAcceleration();
    robotState.rawAccelX = imu.rawAccelX;
    robotState.rawAccelY = imu.rawAccelY;
    robotState.rawAccelZ = imu.rawAccelZ;
    robotState.rawRoll = imu.rawRoll;
    robotState.rawPitch = imu.rawPitch;
    robotState.accelCorrection = accelerationCorrection(
        robotState.forwardAccel, robotState.accelGain,
        MAX_ACCEL_OUTPUT, robotState.pitchDeg);
    robotState.lastImuUpdateMs = millis();
    const bool wheelFresh = robotState.lastMotorFeedbackMs != 0 &&
        millis() - robotState.lastMotorFeedbackMs <= WHEEL_FEEDBACK_STALE_MS;
    robotState.fusedVelocity = velocityEstimator.update(
        robotState.wheelVelocity, robotState.forwardAccel, dtSeconds,
        robotState.wheelDiameterMm, robotState.velocityFusionTau, wheelFresh);

    if (!robotState.enabled) {
        balanceController.reset();
        robotState.velIntegral = 0.0f;
        robotState.targetPitchDeg = 0.0f;
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

    balanceController.setTunings(
        robotState.balanceKp,
        robotState.balanceKi,
        robotState.balanceKd
    );

    balanceController.setDirectionalP(robotState.balanceKp, robotState.balanceKpBack);
    balanceController.setProportionalBoost(
        robotState.balancePBoost, robotState.balancePBoostWidth);
    balanceController.setDerivativeBoost(
        robotState.balanceDBoost, robotState.balanceDWidth);

    robotState.targetPitchDeg = velocityLoop(dtSeconds);

    const BalanceOutput output = balanceController.update(
        robotState.targetPitchDeg,
        robotState.targetAngular,
        robotState.pitchDeg,
        robotState.pitchRateDegPerSec,
        dtSeconds
    );

    const float correctedSpeed = constrain(
        output.speed + FORWARD_MOTOR_SIGN * robotState.accelCorrection,
        -MAX_BALANCE_OUTPUT, MAX_BALANCE_OUTPUT);
    const int16_t speed = normalizedToHoverboard(correctedSpeed);
    const int16_t steer = normalizedToHoverboard(output.steer);

    if (++rlTickPhase >= 2) {
        rlTickPhase = 0;
        const float accelFwd = IMU_FORWARD_ACCEL_SIGN * robotState.rawAccelY;
        const float battery = robotState.batteryVoltage > 1.0f ? robotState.batteryVoltage : 37.0f;
        const float wheelFwd = RL_DRIVE_SIGN * robotState.wheelVelocity;
        if (!rlState.fresh) {
            rlState.u = RL_DRIVE_SIGN * rlAppliedCmd / static_cast<float>(MAX_HOVERBOARD_COMMAND);
        }
        const uint32_t t0 = micros();
        // One call: Arduino's constrain() is a macro that would evaluate (and step) the policy up to 3 times.
        const int16_t rlCmd = static_cast<int16_t>(RL_DRIVE_SIGN * bizbot_policy_tick(
            &rlState, robotState.pitchDeg, accelFwd, wheelFwd, battery));
        rlCmdHeld = constrain(rlCmd, -MAX_HOVERBOARD_COMMAND, MAX_HOVERBOARD_COMMAND);
        const uint32_t us = micros() - t0;
        char line[112];
        const int n = snprintf(line, sizeof(line), "RL %lu %.2f %.3f %.1f %.2f %d %d %lu\n",
            static_cast<unsigned long>(millis()), robotState.pitchDeg, accelFwd, wheelFwd,
            battery, speed, rlCmdHeld, static_cast<unsigned long>(us));
        if (n > 0 && n < static_cast<int>(sizeof(line))) {
            if (wifiControlClient && wifiControlClient.connected()) {
                send(wifiControlClient.fd(), line, n, MSG_DONTWAIT);
            } else if (DebugSerial.availableForWrite() >= n) {
                DebugSerial.write(reinterpret_cast<const uint8_t*>(line), n);
            }
        }
    }
    rlAppliedCmd = rlCmdHeld;
    int16_t rlOut = rlCmdHeld;
    if (rlOut != 0) {
        rlOut = constrain(rlOut + (rlOut > 0 ? RL_DZ_COMP : -RL_DZ_COMP),
                          -MAX_HOVERBOARD_COMMAND, MAX_HOVERBOARD_COMMAND);
    }
    Send(steer, rlOut);

#if PID_DEBUG
    static uint32_t lastPidDebugMs = 0;
    const uint32_t nowMs = millis();
    if (nowMs - lastPidDebugMs >= 500) {
        lastPidDebugMs = nowMs;
        // Send one complete line to the active controller connection.
        char line[256];
        const int length = snprintf(
            line, sizeof(line),
            "PID pitch=%.2f trim=%.2f target=%.2f kp=%.4f ki=%.4f kd=%.4f "
            "p=%.4f i=%.4f d=%.4f out=%.3f cmd=%d accel=%.3f ka=%.3f acorr=%.3f\n",
            robotState.pitchDeg,
            robotState.pitchTrimDeg,
            robotState.targetPitchDeg,
            robotState.balanceKp, robotState.balanceKi, robotState.balanceKd,
            balanceController.lastPTerm(), balanceController.lastITerm(),
            balanceController.lastDTerm(), correctedSpeed, speed,
            robotState.forwardAccel, robotState.accelGain, robotState.accelCorrection
        );
        if (length > 0 && length < static_cast<int>(sizeof(line))) {
            if (wifiControlClient && wifiControlClient.connected()) {
                wifiControlClient.write(
                    reinterpret_cast<const uint8_t*>(line), length);
            } else if (DebugSerial.availableForWrite() >= length) {
                // Skip a diagnostic rather than wait for USB buffer space.
                DebugSerial.write(
                    reinterpret_cast<const uint8_t*>(line), length);
            }
        }
    }
#endif
}

void setup() {
    DebugSerial.setTxBufferSize(512);
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

    robotState.wheelDiameterMm = WHEEL_DIAMETER_MM;
    robotState.balanceKp = BALANCE_KP;
    robotState.balanceKpBack = BALANCE_KP;
    robotState.balanceKi = BALANCE_KI;
    robotState.balanceKd = BALANCE_KD;
    robotState.balancePBoost = BALANCE_P_BOOST_DEG;
    robotState.balancePBoostWidth = BALANCE_P_BOOST_WIDTH_DEG;
    robotState.balanceDBoost = BALANCE_D_BOOST;
    robotState.balanceDWidth = BALANCE_D_WIDTH_DEG;
    robotState.accelGain = ACCEL_GAIN;
    robotState.velKp = VEL_KP;
    robotState.velKi = VEL_KI;
    robotState.velSign = VEL_SIGN;
    robotState.pitchTrimDeg = loadPitchOffset(PITCH_TRIM_DEG);

    robotState.lastCommandMs = millis();
    DebugSerial.println("BizBot firmware ready");
    lastControlUs = micros();
}

void loop() {
    updateCommandInput();
    updateServos();

    if (robotState.enabled && !wasEnabled) {
        onEnabled();
    }
    wasEnabled = robotState.enabled;

    Receive();

    const uint32_t now = micros();
    const uint32_t elapsedUs = now - lastControlUs;
    if (elapsedUs >= CONTROL_PERIOD_US) {
        lastControlUs = now; // Skip missed cycles instead of replaying them.
        if (elapsedUs > 100000 && robotState.enabled) {
            stopRobot("CONTROL_OVERRUN");
        }
        runControlLoop(fminf(elapsedUs * 1e-6f, 0.1f));
    }
}
