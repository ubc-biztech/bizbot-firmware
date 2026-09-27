# BizBot tuner

A browser page for tuning the balance controller with buttons instead of
typing serial commands. It runs on your laptop, talks to the robot over its
Wi-Fi network, and shows a live pitch trace so you can see what a gain change
did.

## Run

```bash
cd tools/tuner
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
