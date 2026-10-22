# BizBot tuner

A browser page for tuning the balance controller with buttons instead of
typing serial commands. It runs on your laptop, talks to the robot over its
Wi-Fi network, and shows a live pitch trace so you can see what a gain change
did.

## Run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cd tuner
npm install          # first time only
npm run dev
```

Join the `BizBot-Control` Wi-Fi network, then open the URL Vite prints
(http://localhost:5173). Close `keyboard_controls.py` first: the firmware
accepts one controller at a time.

The page keeps the STOP keepalive going while it is open. Closing the last tab
sends DISABLE, so walking away always leaves the robot stopped. `Esc` is
DISABLE from anywhere on the page.

To point it at another address (for example a fake robot on localhost):

```bash
BIZBOT_HOST=127.0.0.1 BIZBOT_PORT=4333 npm run dev
```

## How it works

A browser cannot open the robot's raw TCP socket, so `bridge.js` runs inside
the Vite dev server (see `vite.config.js`): one TCP connection to
`192.168.4.1:3333`, every line forwarded to the page over a WebSocket at
`/bridge`, every button click forwarded back as a firmware command. The page
polls `GET_STATE` twice a second and parses the `PID` debug lines when the
firmware is built with `PID_DEBUG=1`.

## Suggested tuning order

1. Hold the robot at what feels like balance and press `Zero IMU`.
2. Raise kp until a push is answered firmly. Fast shivering at rest means
   too far; back off one step.
3. Raise kd if it overshoots after a push. A buzz on the trace means too far.
   Leave ki at 0.
4. Save a preset when something works. Gains reset on reboot, so copy the
   ones you keep into `src/config.h`.

The page only sends commands the firmware on `main` understands: ENABLE,
DISABLE, STOP, GET_STATE, SET_PID, SET_TRIM, ZERO_IMU.

### Near-zero P boost

The Pitch loop card includes a live Boost control (0–2°, default 0.65°). Zero disables the boost; Width is adjustable from 0.1–5° (default 0.5°) with `SET_P_WIDTH <degrees>`; `GET_STATE` reports `pwidth`. Smaller widths concentrate and strengthen the response near zero; this is a smooth transition, not a hard cutoff. It sends `SET_P_BOOST <degrees>`, and `GET_STATE` reports `pboost`. Values are saved in browser presets; firmware restores boost and width defaults on reboot. The Live section shows the active robot value.

### Large-angle D boost

The Pitch loop card has D Boost (0–4) and Angle (0.1–35°) controls. The live command is `SET_D_BOOST <boost> <angle>`, with `dboost` and `dwidth` in `GET_STATE`. Both values are included in presets. Reboot defaults are Boost 0 (off), Angle 5°.

For angle error `e`, effective D is `kd * (1 + boost * e² / (e² + angle²))`. D remains based on negative measured pitch rate. At zero error it equals base D; at the selected angle half the extra gain applies; far beyond that angle it approaches `kd * (1 + boost)`. Positive and negative errors behave identically. The angle is relative to the trimmed target, not raw IMU pitch.

### Front/back P

P front and P back independently set proportional gain on the positive and negative sides of the trimmed target pitch. Both start at 0.04. The shared P boost scales whichever gain applies; I and D remain shared. `SET_PID_FB <front> <back> <ki> <kd>` updates all four gains atomically. Legacy `SET_PID` sets both P gains equally. `GET_STATE` exposes `kp_front` and `kp_back` (`kp` remains an alias for front). Older browser settings and presets use their existing P for both directions. Firmware restores defaults on reboot.

### Velocity sensor fusion

`SET_FUSION <wheel diameter mm> <time constant seconds>` configures the complementary velocity estimator. Diameter must be 50–500 mm, or 0 to leave fusion unconfigured. Time constant is 0–1 s; 0 selects wheel-only feedback. Defaults: measured diameter 160 mm, time constant 0.2 s. Use the measured tire outer diameter. Apply the browser settings again after firmware reboot.

The estimator integrates existing gravity-compensated, filtered forward acceleration, with continuous correction toward signed mean wheel RPM. The velocity PI retains its RPM-based gains and receives the fused estimate. Larger time constants give acceleration more influence but allow more bias error. This reduces integration drift; it does not eliminate errors from sensor bias, wheel slip, or body rotation. Verify forward acceleration and wheel-speed signs agree before tuning velocity PI. Existing direct acceleration correction is separate and remains at its current setting.

`GET_STATE` includes wheel `velocity`, `fused_velocity`, `wheel_mm`, and `fusion_tau`. Wheel feedback older than 200 ms resets the estimator and retains the existing neutral-return behavior; IMU faults retain the existing stop behavior. PI gains still default to zero.

Host checks: `c++ -std=c++11 test/velocity/test_estimator.cpp -o /tmp/bizbot-velocity-estimator-test && /tmp/bizbot-velocity-estimator-test`.
