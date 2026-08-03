# IMU Flashing & Testing Guide (BNO085 / GY-BM008X)

End-to-end steps to wire, flash, and verify the BNO085 IMU driver on the ESP32.

The BNO085 is a fusion IMU: it computes orientation onboard and streams a fused
quaternion over I2C. The firmware converts that quaternion to a pitch angle and
reads the calibrated gyro for pitch rate.

---

## 1. Wiring

### IMU — BNO085 (GY-BM008X), UART-RVC

The BNO08x is driven over **UART-RVC, not I2C**. Its I2C implementation violates
the I2C spec and wedges the ESP32's I2C peripheral (`requestFrom Error 263`, dead
bus) — a documented ESP32/BNO08x incompatibility. UART-RVC streams pitch/roll/yaw
at ~100 Hz over a single wire and is the simplest reliable path on ESP32.

| Module pin | ESP32       | Notes                                        |
|------------|-------------|----------------------------------------------|
| VCC        | 3V3         | Do **not** use 5V unless the board regulates |
| GND        | GND         | Shared ground is required                    |
| **SDA**    | **GPIO 32** | In RVC mode SDA is the sensor's data-out (its "TX") → ESP32 RX (`IMU_UART_RX_PIN`, UART1) |
| PS mode a  | **3V3**     | datasheet PS0 — often labeled **P0 / PS0 / PS1** |
| PS mode b  | **GND**     | datasheet PS1 — often labeled **P1 / PS1 / PS2** |
| SCL, ADO, CS, INT, RST | *unconnected* | not needed for RVC (tie RST→3V3 if flaky) |

> There is **no pin literally labeled TX** — in UART-RVC mode the multiplexed
> **SDA** pin carries the sensor's serial output. Wire only that; RVC is one-way.
>
> **Mode pins:** the two protocol-select pins must be at **opposite levels** for
> RVC (datasheet: PS0=HIGH, PS1=LOW). Board silkscreens vary — some print
> `P0/P1`, some `PS0/PS1`, some `PS1/PS2`. If `FAULT IMU_INIT`, **swap the two
> levels** — with only two pins there are only two orderings to try.
>
> UART2 (Serial2) is reserved for the hoverboard, so the IMU uses **UART1
> (Serial1)** on GPIO 32.

> **Note:** UART-RVC provides no gyro output, so pitch *rate* is derived by
> differentiating pitch (see `IMU_RATE_LPF_ALPHA` in `config.h`). It's noisier
> than a true gyro — expect to use a smaller `BALANCE_KD` when tuning.

### Hoverboard UART (not needed for IMU testing — reference only)

| Mainboard        | ESP32       |
|------------------|-------------|
| PB10 (TX, blue)  | GPIO 25 (RX)|
| PB11 (RX, green) | GPIO 26 (TX)|
| GND (black)      | GND         |
| 15V (red)        | **DO NOT CONNECT** |

---

## 2. One-time setup

`pio` may not be on your PATH. Either add it once:

```bash
echo 'export PATH="$HOME/.platformio/penv/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc
```

...or prefix every command with `~/.platformio/penv/bin/`.

Find the ESP32's serial port after plugging it in:

```bash
pio device list
```

Look for a new `/dev/cu.usbserial-*`, `/dev/cu.wchusbserial-*`, or
`/dev/cu.SLAB_USBtoUART`. If nothing new appears, install the USB-UART driver
for your board (CP210x for Silicon Labs, CH340 for WCH) and re-plug.

---

## 3. Flash and open the serial monitor

Build, upload, and open the monitor in one line (PlatformIO auto-detects the port):

```bash
pio run --target upload && pio device monitor
```

- Baud is already 115200 (`monitor_speed` in `platformio.ini`).
- **Exit the monitor with `Ctrl+C`.**
- Only one program can hold the port — **close the monitor before re-flashing**,
  or the upload fails with a "port busy" error.
- If upload stalls at "Connecting…", hold the **BOOT** button while it connects.

To pin the port explicitly:

```bash
pio run --target upload --upload-port /dev/cu.usbserial-XXXX
pio device monitor --port /dev/cu.usbserial-XXXX
```

### Expected boot output

```
OK IMU_INIT
BizBot firmware ready
```

- `OK IMU_INIT` → the BNO085 answered on I2C. Good.
- `FAULT IMU_INIT` → not detected. Check PS0/PS1 = GND, SDA/SCL wiring, and 3V3/GND.

---

## 4. Step A — IMU orientation calibration

The robot's "pitch" axis depends on how the board is physically mounted, so you
must confirm which Euler angle and gyro axis correspond to leaning forward.

1. Edit `platformio.ini` and enable the debug dump — set the last line to:

   ```ini
   build_flags = -DIMU_DEBUG_EULER=1
   ```

2. Re-flash and open the monitor:

   ```bash
   pio run --target upload && pio device monitor
   ```

3. You'll see a stream every ~200 ms:

   ```
   IMU rpy(deg)=<roll>,<pitch>,<yaw> gyro(dps)=<gx>,<gy>,<gz>
   ```

4. With the robot **level**, note the resting values. Then **tip it forward**
   (the direction it would fall while driving) and observe:
   - Which of roll / pitch / yaw **changes the most** → your `IMU_PITCH_SOURCE`.
   - Which gyro axis (gx/gy/gz) **spikes** → your `IMU_RATE_SOURCE`.
   - The **sign** while leaning forward — it must be **positive**.

5. Set the four constants in `src/config.h`:

   ```cpp
   constexpr int   IMU_PITCH_SOURCE = IMU_PITCH_FROM_PITCH; // or _ROLL / _YAW
   constexpr float IMU_PITCH_SIGN   = 1.0f;                 // -1.0f to flip
   constexpr int   IMU_RATE_SOURCE  = IMU_RATE_FROM_GYRO_Y; // or _GYRO_X / _GYRO_Z
   constexpr float IMU_RATE_SIGN    = 1.0f;                 // -1.0f to flip
   ```

   Rule: **leaning forward must produce a positive `pitchDeg` and a positive
   pitch rate.** Flip the corresponding `*_SIGN` to `-1.0f` if it reads negative.

6. **Re-comment the debug flag** in `platformio.ini` (back to `; build_flags = ...`)
   and re-flash for normal operation.

### Verify with GET_STATE (optional)

After calibration, with the debug flag off, you can confirm the selected pitch
via the command protocol. In the serial monitor, type:

```
GET_STATE
```

You'll get a line like:

```
STATE enabled=0 linear=0.000 angular=0.000 pitch=<deg> pitch_rate=<dps> battery=... wheel_l=0 wheel_r=0
```

Tilting the robot forward should drive `pitch` positive.

---

## 5. Safety notes before any motor test

- Keep the **wheels off the ground** and the **chassis restrained** for first tests.
- Never connect the hoverboard **15V** line to the ESP32.
- Share ground between ESP32 and hoverboard controller.
- The balance PID gains in `config.h` are **placeholders** — expect untuned
  behavior until tuned on restrained hardware.
- Safety already built in: command timeout (500 ms), tilt limit (35°), and a
  latched stop if the IMU stops reporting for >100 ms.
