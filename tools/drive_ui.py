"""Drive BizBot from a browser page: W/S sets the RL policy's target wheel speed, A/D steers.

    .venv/bin/python -u tools/drive_ui.py --transport wifi      (what `make drive` runs)

Owns the robot connection (same transports as keyboard_controls.py), sends ENABLE on connect and
keeps the link alive with DRIVE/STOP only, and serves the page in drive_ui.html. Every robot line is printed as
`< line`, so piping through tee gives the same log as keyboard_controls.py. Ctrl-C sends DISABLE.
The page opens by itself unless a tab from an earlier run is still open (it reconnects).
"""

import argparse
import json
import math
import sys
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import serial

from keyboard_controls import DEFAULT_WIFI_HOST, DEFAULT_WIFI_PORT, open_transport

SEND_INTERVAL_S = 0.2      # DRIVE/STOP keepalive, as keyboard_controls.py; the firmware disables after 500 ms
INPUT_EXPIRY_S = 0.35      # held keys count only while the page keeps repeating them
MAX_TARGET_RPM = 150.0     # firmware limits (config.h)
MAX_TURN_COMMAND = 200.0
QUIET_REPLIES = ("OK STOP", "OK DRIVE")


class Robot:
    def __init__(self, transport):
        self.transport = transport
        self.running = threading.Event()
        self.running.set()
        self.lock = threading.Lock()
        self.write_lock = threading.Lock()
        self.target_rpm = 40.0       # W/S target while held
        self.turn_cmd = 40.0         # A/D steer while held
        self.direction = 0
        self.turn = 0
        self.sequence = -1
        self.input_at = 0.0
        self.sent = (0.0, 0.0)       # target rpm and steer in the last keepalive
        self.rl = None               # latest RL line: pitch, wheel rpm, battery, command, policy us
        self.rl_at = 0.0
        self.enabled = False         # from the OK ENABLED / OK DISABLED / ERR DISABLED replies
        self.log = deque(maxlen=40)
        self.log_seq = 0
        self.page_seen = 0.0

    def note(self, text):
        with self.lock:
            if self.log and self.log[-1][1] == text:
                return
            self.log_seq += 1
            self.log.append((self.log_seq, text))

    def send(self, command, echo=True):
        if echo:
            print(">", command, flush=True)
            self.note("> " + command)
        with self.write_lock:
            self.transport.send_line(command)

    def inputs(self):
        with self.lock:
            live = time.monotonic() - self.input_at <= INPUT_EXPIRY_S
            return (self.direction, self.turn) if live else (0, 0)

    def keepalive_loop(self):
        last = None
        while self.running.is_set():
            direction, turn = self.inputs()
            with self.lock:
                target, steer = direction * self.target_rpm, turn * self.turn_cmd
            command = f"DRIVE {target:.1f} {steer:.1f}" if target or steer else "STOP"
            if command != last:
                print(f"> {command}  @{time.time():.3f}", flush=True)   # logged once per change, for the drive analysis
                last = command
            try:
                self.send(command, echo=False)
            except (OSError, serial.SerialException) as error:
                print(f"Connection lost while sending: {error}", file=sys.stderr, flush=True)
                self.running.clear()
                return
            with self.lock:
                self.sent = (target, steer)
            time.sleep(SEND_INTERVAL_S)

    def receive_loop(self):
        while self.running.is_set():
            try:
                line = self.transport.read_line()
            except (OSError, serial.SerialException, ConnectionError) as error:
                print(f"Connection lost while receiving: {error}", file=sys.stderr, flush=True)
                self.running.clear()
                return
            line = line.strip() if line else ""
            if not line or line in QUIET_REPLIES:
                continue
            print("<", line, flush=True)
            if line.startswith("RL "):
                self.read_rl(line)
                continue
            if line == "OK ENABLED":
                self.enabled = True
            elif line in ("OK DISABLED", "ERR DISABLED"):
                self.enabled = False
            self.note(line)

    def read_rl(self, line):
        # RL <ms> <pitch_deg> <accel_fwd> <wheel_rpm> <battery_v> <pid_cmd> <rl_cmd> <us>
        try:
            v = [float(x) for x in line.split()[1:]]
            rl = {"pitch": v[1], "wheel_rpm": v[3], "battery": v[4], "command": v[6], "policy_us": v[7]}
        except (ValueError, IndexError):
            return
        with self.lock:
            self.rl, self.rl_at = rl, time.monotonic()

    def is_enabled(self):
        return self.enabled or time.monotonic() - self.rl_at <= 0.5

    def snapshot(self):
        direction, turn = self.inputs()
        with self.lock:
            enabled = self.is_enabled()
            rl = self.rl if time.monotonic() - self.rl_at <= 0.5 else None   # RL lines stop when disabled
            pitch = rl["pitch"] if rl else None
            rpm = rl["wheel_rpm"] if rl else None
            diameter_m = 0.160
            target, steer = self.sent
            status = ("Disabled" if not enabled else
                      "Driving" if direction else "Turning" if turn else "Balancing")
            return {
                "connected": self.running.is_set(), "link": self.transport.description,
                "enabled": enabled, "status": status,
                "target_rpm": self.target_rpm, "turn_cmd": self.turn_cmd,
                "target": target, "steer": steer,
                "pitch": pitch, "wheel_rpm": rpm,
                "speed": rpm / 60.0 * math.pi * diameter_m if rpm is not None else None,
                "target_speed": target / 60.0 * math.pi * diameter_m,
                "battery": rl["battery"] if rl else None,
                "command": rl["command"] if rl else None,
                "policy_us": rl["policy_us"] if rl else None,
                "log": list(self.log),
            }

    def command(self, data):
        name = data["name"]
        if name == "drive":
            direction, turn, sequence = data["direction"], data.get("turn", 0), data["sequence"]
            if type(direction) is not int or direction not in (-1, 0, 1):
                raise ValueError("Invalid direction")
            if type(turn) is not int or turn not in (-1, 0, 1):
                raise ValueError("Invalid turn")
            if type(sequence) is not int or not 0 <= sequence < 2**53:
                raise ValueError("Invalid sequence")
            with self.lock:
                if sequence > self.sequence:
                    self.direction, self.turn, self.sequence = direction, turn, sequence
                    self.input_at = time.monotonic()
        elif name in ("enable", "disable"):
            with self.lock:
                self.direction = self.turn = 0
            self.send(name.upper())
        elif name in ("target_rpm", "turn_cmd"):
            value = float(data["value"])
            limit = MAX_TARGET_RPM if name == "target_rpm" else MAX_TURN_COMMAND
            if not math.isfinite(value) or not 0 <= value <= limit:
                raise ValueError("Invalid value")
            with self.lock:
                setattr(self, name, value)
        else:
            raise ValueError("Unknown command")


def serve(robot, ui_port):
    page = Path(__file__).with_name("drive_ui.html").read_bytes()
    url = f"http://127.0.0.1:{ui_port}"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, body, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self.reply(200, page, "text/html; charset=utf-8")
            elif self.path == "/state":
                robot.page_seen = time.monotonic()
                self.reply(200, json.dumps(robot.snapshot()).encode())
            else:
                self.reply(404, b"{}")

        def do_POST(self):
            if self.path != "/command":
                self.reply(404, b"{}")
                return
            if self.headers.get("Origin") not in (None, url, f"http://localhost:{ui_port}"):
                self.reply(403, b"{}")
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1024:
                    raise ValueError("Invalid request size")
                robot.command(json.loads(self.rfile.read(size)))
            except (ValueError, TypeError, KeyError, AttributeError):
                self.reply(400, b'{"error":"Invalid control value"}')
                return
            except (OSError, serial.SerialException):
                self.reply(503, b'{"error":"Robot connection lost"}')
                return
            self.reply(202, b"{}")

    server = ThreadingHTTPServer(("127.0.0.1", ui_port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, url


def parse_args():
    parser = argparse.ArgumentParser(description="Browser drive page for BizBot.")
    parser.add_argument("--transport", choices=("serial", "wifi"), default="serial",
                        help="control connection to use (default: serial)")
    parser.add_argument("--serial-port", help="USB serial device; auto-detected when omitted")
    parser.add_argument("--host", default=DEFAULT_WIFI_HOST,
                        help=f"ESP32 Wi-Fi IP address (default: {DEFAULT_WIFI_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_WIFI_PORT,
                        help=f"ESP32 TCP control port (default: {DEFAULT_WIFI_PORT})")
    parser.add_argument("--ui-port", type=int, default=8766, help="port the page is served on (default: 8766)")
    parser.add_argument("--no-open", action="store_true", help="never open the browser")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        robot = Robot(open_transport(args))
    except (OSError, serial.SerialException) as error:
        print(f"Connection error: {error}", file=sys.stderr)
        print("Check the USB port or connect to the BizBot-Control Wi-Fi network.", file=sys.stderr)
        sys.exit(1)
    try:
        server, url = serve(robot, args.ui_port)
    except OSError as error:
        robot.transport.close()
        print(f"Cannot serve the page on port {args.ui_port}: {error}. Is another drive page running?", file=sys.stderr)
        sys.exit(1)
    try:
        robot.send("ENABLE")
        threads = [threading.Thread(target=f, daemon=True) for f in (robot.keepalive_loop, robot.receive_loop)]
        for t in threads:
            t.start()
        print(f"Drive page: {url}  (Ctrl-C disables balance and quits)", flush=True)
        if not args.no_open:
            time.sleep(1.5)   # a tab left open from an earlier run reconnects within this time
            if not robot.page_seen:
                webbrowser.open(url)
        while robot.running.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        robot.running.clear()
        for command in ("DISABLE", "STOP", "DISABLE"):
            try:
                robot.send(command)
            except (OSError, serial.SerialException):
                break
        server.shutdown()
        robot.transport.close()
        print("Disconnected.")


if __name__ == "__main__":
    main()
