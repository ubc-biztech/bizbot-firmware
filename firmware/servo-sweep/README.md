# Standalone ESP32 three-servo sweep

This is an independent PlatformIO project with its own configuration and main.cpp.
It does not compile or use BizBot firmware, IMU, Wi-Fi, command parsing, or wheel
control. Uses Arduino and ESP32Servo 1.1.1 (compatible with the installed ESP32 Arduino 2.x core). It does not communicate with the
hoverboard; disconnect/turn off wheel hardware during this servo-only test.

Signals: servo 1 GPIO18, servo 2 GPIO19, servo 3 GPIO23.
Use an external servo-rated power supply and connect its ground to ESP32 GND.

Three seconds after boot, each servo sweeps nominal 0 -> 180 -> 0 once, in order.
Each sweep takes about 15 seconds. Signals then remain LOW. Press EN/Reset to repeat.
Angles are set with Servo.write(angle); ESP32Servo generates the 50 Hz signal. The default pulse
range is 1000–2000 microseconds; actual travel depends on the servo. Change
MIN_PULSE_US / MAX_PULSE_US in src/main.cpp only according to the servo specification.
Keep mechanisms clear: motion begins automatically after flashing or resetting.

From the repository root:

```sh
~/.platformio/penv/bin/pio run -d standalone/servo-sweep
~/.platformio/penv/bin/pio run -d standalone/servo-sweep -t upload --upload-port /dev/cu.usbserial-10
```

To restore BizBot, upload the root project's esp32dev environment.
