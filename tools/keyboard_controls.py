"""Hold W/A/S/D to drive BizBot over USB serial or local Wi-Fi.

Setup:
    pip install pyserial pynput

macOS: System Settings → Privacy & Security → grant Accessibility AND
Input Monitoring to the app hosting your terminal. No sudo needed.
"""

import argparse
from collections import deque
import socket
import sys
import threading
import time

import serial
from pynput import keyboard as pynput_keyboard
from serial.tools import list_ports

BAUD = 115200
DEFAULT_WIFI_HOST = "192.168.4.1"
DEFAULT_WIFI_PORT = 3333
KEEPALIVE_INTERVAL_S = 0.2
POLL_INTERVAL_S = 0.05

LINEAR_SPEED = 0.25
ANGULAR_SPEED = 0.25

running = threading.Event()
running.set()

keepalive_cmd = "STOP"
keepalive_lock = threading.Lock()
write_lock = threading.Lock()
console_lock = threading.Lock()
console_paused = False
deferred_messages = deque(maxlen=32)


def background_print(*args, telemetry=False, **kwargs):
    """Drain telemetry silently during input; defer command replies/errors."""
    with console_lock:
        if console_paused:
            if not telemetry:
                deferred_messages.append((args, kwargs))
            return
        print(*args, **kwargs)


def pause_console():
    global console_paused
    with console_lock:
        console_paused = True


def resume_console():
    global console_paused
    with console_lock:
        console_paused = False
        while deferred_messages:
            args, kwargs = deferred_messages.popleft()
            print(*args, **kwargs)

# Characters currently held down, maintained by the pynput listener thread.
held_keys = set()
held_lock = threading.Lock()


def _key_char(key):
    """Return the lowercase character for a key press, or None for specials."""
    try:
        char = key.char
    except AttributeError:
        return None
    return char.lower() if char else None


def _on_press(key):
    char = _key_char(key)
    if char:
        with held_lock:
            held_keys.add(char)


def _on_release(key):
    char = _key_char(key)
    if char:
        with held_lock:
            held_keys.discard(char)


def is_held(char):
    with held_lock:
        return char in held_keys


class SerialTransport:
    def __init__(self, port):
        self.port = port
        self.connection = serial.Serial(port, BAUD, timeout=0.1)

    @property
    def description(self):
        return f"USB serial {self.port} at {BAUD} baud"

    def send_line(self, command):
        self.connection.write((command + "\n").encode())

    def read_line(self):
        data = self.connection.readline()
        if not data:
            return None
        return data.decode(errors="replace").strip()

    def close(self):
        self.connection.close()


class WifiTransport:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.connection = socket.create_connection((host, port), timeout=3)
        self.connection.settimeout(0.1)
        self.read_buffer = bytearray()

    @property
    def description(self):
        return f"Wi-Fi TCP {self.host}:{self.port}"

    def send_line(self, command):
        self.connection.sendall((command + "\n").encode())

    def read_line(self):
        newline_index = self.read_buffer.find(b"\n")

        if newline_index < 0:
            try:
                data = self.connection.recv(256)
            except socket.timeout:
                return None

            if not data:
                raise ConnectionError("the ESP32 closed the Wi-Fi connection")

            self.read_buffer.extend(data)
            newline_index = self.read_buffer.find(b"\n")

        if newline_index < 0:
            return None

        line = bytes(self.read_buffer[:newline_index])
        del self.read_buffer[:newline_index + 1]
        return line.decode(errors="replace").rstrip("\r")

    def close(self):
        self.connection.close()


def find_serial_port():
    """Pick the first likely USB serial device, or fall back to a manual default."""
    keywords = ("usb", "acm", "uart", "serial")

    for port in list_ports.comports():
        device = port.device.lower()
        description = port.description.lower()

        if any(keyword in device or keyword in description for keyword in keywords):
            return port.device

    if sys.platform == "darwin":
        return "/dev/cu.usbserial-0001"
    if sys.platform.startswith("linux"):
        return "/dev/ttyUSB0"
    return "COM3"


def send(transport, cmd, echo=True):
    if echo:
        background_print(">", cmd)

    with write_lock:
        transport.send_line(cmd)


def set_keepalive(cmd):
    global keepalive_cmd

    with keepalive_lock:
        keepalive_cmd = cmd


def get_keepalive():
    with keepalive_lock:
        return keepalive_cmd


def keepalive_loop(transport):
    while running.is_set():
        try:
            # Send repeatedly, not only when the command changes.
            send(transport, get_keepalive(), echo=False)
        except (OSError, serial.SerialException) as error:
            background_print(f"Connection lost while sending: {error}", file=sys.stderr)
            running.clear()
            return

        time.sleep(KEEPALIVE_INTERVAL_S)


def receive_loop(transport):
    while running.is_set():
        try:
            line = transport.read_line()
        except (OSError, serial.SerialException) as error:
            background_print(f"Connection lost while receiving: {error}", file=sys.stderr)
            running.clear()
            return

        if line and line.strip() != "OK STOP":
            background_print("<", line, telemetry=line.startswith(("PID ", "STATE ")))


def movement_command():
    """Return the command for currently held WASD keys, or STOP if none."""
    forward = is_held("w")
    backward = is_held("s")
    left = is_held("a")
    right = is_held("d")

    linear = 0.0
    angular = 0.0

    if forward and not backward:
        linear = LINEAR_SPEED
    elif backward and not forward:
        linear = -LINEAR_SPEED

    if left and not right:
        angular = ANGULAR_SPEED
    elif right and not left:
        angular = -ANGULAR_SPEED

    if linear == 0.0 and angular == 0.0:
        return "STOP"

    return f"CMD_VEL {linear} {angular}"


def handle_tuning_input(transport):
    """Prompt for PID gains and send SET_PID; movement pauses while typing."""
    # Clear movement requests while we type; balancing remains active.
    # The keepalive thread keeps
    # sending STOP in the background, so the firmware link never times out.
    set_keepalive("STOP")

    pause_console()
    try:
        # Drop the 't' keystrokes (and anything else) already sitting in the
        # terminal input buffer so the prompt starts empty.
        if sys.platform != "win32":
            import termios
            time.sleep(0.2)  # give the held key time to be released
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
    
        print()
        raw = input("kp ki kd, 'zero', 'offset <deg>', or 'accel <gain>' (blank to cancel): ").strip()
    finally:
        resume_console()

    if not running.is_set():
        return

    if not raw:
        print("Tuning cancelled.")
        return

    parts = raw.split()

    if raw.lower() == "zero":
        send(transport, "ZERO_IMU")
    elif parts[0].lower() == "accel":
        if len(parts) != 2:
            print("Usage: accel <gain>; accel 0 disables acceleration feedback")
            return
        try:
            gain = float(parts[1])
        except ValueError:
            print("Acceleration gain must be a number")
            return
        if not 0.0 <= gain <= 0.1:
            print("Acceleration gain must be between 0 and 0.1")
            return
        send(transport, f"SET_ACCEL {gain}")
    elif parts[0].lower() in ("trim", "offset"):
        if len(parts) != 2:
            print("Usage: offset <degrees>, e.g.: offset -2.5 (sets the absolute offset)")
            return
        try:
            trim = float(parts[1])
        except ValueError:
            print(f"Not a number: {parts[1]}")
            return
        if not -15.0 <= trim <= 15.0:
            print("Offset must be between -15 and 15 degrees")
            return
        send(transport, f"SET_TRIM {trim}")
    else:
        if len(parts) != 3:
            print("Need kp ki kd, 'zero', 'offset <deg>', or 'accel <gain>'")
            return
        try:
            kp, ki, kd = (float(part) for part in parts)
        except ValueError:
            print(f"Not numbers: {raw}")
            return
        send(transport, f"SET_PID {kp} {ki} {kd}")

    time.sleep(0.3)  # let the response print and the T key release


def stdin_command_loop(transport):
    """Forward raw command lines from a piped stdin (tools/tuner bridge)."""
    for raw in sys.stdin:
        if not running.is_set():
            return
        line = raw.strip()
        if line:
            try:
                send(transport, line)
            except (OSError, serial.SerialException) as error:
                background_print(f"Connection lost while sending: {error}", file=sys.stderr)
                running.clear()
                return
    running.clear()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Drive BizBot using USB serial or its local Wi-Fi network."
    )
    parser.add_argument(
        "--transport",
        choices=("serial", "wifi"),
        default="serial",
        help="control connection to use (default: serial)",
    )
    parser.add_argument(
        "--serial-port",
        help="USB serial device; auto-detected when omitted",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_WIFI_HOST,
        help=f"ESP32 Wi-Fi IP address (default: {DEFAULT_WIFI_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_WIFI_PORT,
        help=f"ESP32 TCP control port (default: {DEFAULT_WIFI_PORT})",
    )
    return parser.parse_args()


def open_transport(args):
    if args.transport == "wifi":
        transport = WifiTransport(args.host, args.port)
    else:
        port = args.serial_port or find_serial_port()
        transport = SerialTransport(port)
        time.sleep(2)  # wait for the ESP32 reset after opening USB serial

    print(f"Connected using {transport.description}")
    return transport


def main():
    args = parse_args()
    transport = open_transport(args)
    keepalive_thread = None
    receiver_thread = None

    running.set()

    def on_press(key):
        if _key_char(key) == "q":
            # Handle this outside the main loop so Q also disables balance
            # while the tuning prompt is blocked waiting for input.
            set_keepalive("DISABLE")
            running.clear()
            try:
                send(transport, "DISABLE")
                background_print("Balance disable requested.")
            except (OSError, serial.SerialException) as error:
                background_print(f"Could not send DISABLE: {error}", file=sys.stderr)
            return
        _on_press(key)

    listener = pynput_keyboard.Listener(on_press=on_press, on_release=_on_release)

    try:
        send(transport, "ENABLE")
        set_keepalive("STOP")
        listener.start()

        keepalive_thread = threading.Thread(
            target=keepalive_loop,
            args=(transport,),
            daemon=True,
        )
        keepalive_thread.start()

        receiver_thread = threading.Thread(
            target=receive_loop,
            args=(transport,),
            daemon=True,
        )
        receiver_thread.start()

        if not sys.stdin.isatty():
            threading.Thread(target=stdin_command_loop, args=(transport,), daemon=True).start()

        print(
            "Hold W/A/S/D to move. Release to stop. "
            "I = state. T = tune PID. Q = disable balance and quit."
        )

        while running.is_set():
            if is_held("q"):
                break

            if is_held("t") and sys.stdin.isatty():
                handle_tuning_input(transport)
            elif is_held("i"):
                set_keepalive("STOP")
                send(transport, "GET_STATE")
                time.sleep(0.2)
            else:
                set_keepalive(movement_command())

            time.sleep(POLL_INTERVAL_S)

    finally:
        running.clear()
        listener.stop()

        # Disable before waiting for worker threads to finish.
        try:
            send(transport, "DISABLE")
        except (OSError, serial.SerialException):
            pass

        if keepalive_thread:
            keepalive_thread.join(timeout=1)
        if receiver_thread:
            receiver_thread.join(timeout=1)

        try:
            send(transport, "STOP")
            send(transport, "DISABLE")
        except (OSError, serial.SerialException):
            pass
        finally:
            transport.close()
            print("Disconnected.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, serial.SerialException) as error:
        print(f"Connection error: {error}", file=sys.stderr)
        print("Check the USB port or connect to the BizBot-Control Wi-Fi network.", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
