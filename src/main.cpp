/*
 * ESP32 <-> Hoverboard bring-up sketch
 * For EFeru hoverboard-firmware-hack-FOC, VARIANT_USART on USART3.
 *
 * STAGE 0b: verify the serial link + feedback + a single wheel responding.
 * This is NOT the balance loop. Get this working first.
 *
 * Wiring (Right Sideboard connector = USART3, 5V tolerant):
 *   Mainboard PB10 (TX,  blue)  -> ESP32 RX  (HOVER_RX_PIN)
 *   Mainboard PB11 (RX,  green) -> ESP32 TX  (HOVER_TX_PIN)
 *   Mainboard GND (black)       -> ESP32 GND
 *   Mainboard 15V (red)         -> DO NOT CONNECT
 */

#include <Arduino.h>

// ----------------------------- Config -----------------------------
#define HOVER_SERIAL_BAUD  115200    // must match the firmware's HOVER_SERIAL_BAUD
#define START_FRAME        0xABCD    // frame marker shared by both sides
#define TIME_SEND          100       // ms between commands; you MUST keep sending or the board times out and zeroes the command
#define HOVER_RX_PIN       25        // ESP32 pin wired to mainboard TX (PB10)
#define HOVER_TX_PIN       26        // ESP32 pin wired to mainboard RX (PB11)

// SAFETY: start at 0 (link test only, no motion).
// Raise to ~30-50 ONLY with the wheels off the ground.
#define TEST_SPEED         0
#define TEST_STEER         0

HardwareSerial &HoverSerial = Serial2;   // UART2 to the mainboard
HardwareSerial &DebugSerial = Serial;    // USB serial monitor

// --------------------------- Protocol ----------------------------
typedef struct {
  uint16_t start;
  int16_t  steer;
  int16_t  speed;
  uint16_t checksum;
} SerialCommand;

typedef struct {
  uint16_t start;
  int16_t  cmd1;
  int16_t  cmd2;
  int16_t  speedR_meas;
  int16_t  speedL_meas;
  int16_t  batVoltage;
  int16_t  boardTemp;
  uint16_t cmdLed;
  uint16_t checksum;
} SerialFeedback;

SerialCommand  Command;
SerialFeedback Feedback;
SerialFeedback NewFeedback;

// ----------------------------- Send ------------------------------
void Send(int16_t steer, int16_t speed) {
  Command.start    = (uint16_t)START_FRAME;
  Command.steer    = steer;
  Command.speed    = speed;
  Command.checksum = (uint16_t)(Command.start ^ Command.steer ^ Command.speed);
  HoverSerial.write((uint8_t *)&Command, sizeof(Command));
}

// --------------------- Receive (start-frame sync) -----------------
void Receive() {
  static uint8_t  idx  = 0;
  static uint8_t  prev = 0;
  static uint8_t *p    = (uint8_t *)&NewFeedback;

  while (HoverSerial.available()) {
    uint8_t  in    = HoverSerial.read();
    uint16_t frame = ((uint16_t)in << 8) | prev;   // bytes arrive low-then-high

    if (frame == START_FRAME) {                     // (re)sync on a fresh frame
      p = (uint8_t *)&NewFeedback;
      *p++ = prev;
      *p++ = in;
      idx = 2;
    } else if (idx >= 2 && idx < sizeof(SerialFeedback)) {
      *p++ = in;
      idx++;
    }

    if (idx == sizeof(SerialFeedback)) {            // full frame collected
      uint16_t checksum = (uint16_t)(
          NewFeedback.start ^ NewFeedback.cmd1 ^ NewFeedback.cmd2 ^
          NewFeedback.speedR_meas ^ NewFeedback.speedL_meas ^
          NewFeedback.batVoltage ^ NewFeedback.boardTemp ^ NewFeedback.cmdLed);

      if (NewFeedback.start == START_FRAME && checksum == NewFeedback.checksum) {
        Feedback = NewFeedback;                     // good frame
        DebugSerial.printf(
          "OK  spdR:%-5d spdL:%-5d  batt:%-5d (~%.1fV)  temp:%d\n",
          Feedback.speedR_meas, Feedback.speedL_meas,
          Feedback.batVoltage, Feedback.batVoltage / 100.0,
          Feedback.boardTemp);
      } else {
        DebugSerial.println("Frame received but checksum failed");
      }
      idx = 0;
    }
    prev = in;
  }
}

// ----------------------------- Main ------------------------------
unsigned long lastSend = 0;

void setup() {
  DebugSerial.begin(115200);
  HoverSerial.begin(HOVER_SERIAL_BAUD, SERIAL_8N1, HOVER_RX_PIN, HOVER_TX_PIN);
  DebugSerial.println("ESP32 hoverboard bring-up starting...");
}

void loop() {
  Receive();                                  // drain feedback every pass

  unsigned long now = millis();
  if (now - lastSend >= TIME_SEND) {
    lastSend = now;
    Send(TEST_STEER, TEST_SPEED);             // keep the command alive
  }
}