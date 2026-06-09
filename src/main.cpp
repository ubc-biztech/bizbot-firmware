#include <Arduino.h>

constexpr unsigned long HEARTBEAT_INTERVAL_MS = 1000;
unsigned long lastHeartbeatMs = 0;

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(50);

  delay(1000);
  Serial.println();
  Serial.println("BizBot ESP32 firmware started");
  Serial.println("Type a line and press Enter to test USB serial.");
}

void loop() {
  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    line.trim();

    if (!line.isEmpty()) {
      Serial.print("ESP32 received: ");
      Serial.println(line);
    }
  }

  const unsigned long now = millis();
  if (now - lastHeartbeatMs >= HEARTBEAT_INTERVAL_MS) {
    lastHeartbeatMs = now;
    Serial.print("HEARTBEAT uptime_ms=");
    Serial.println(now);
  }
}
