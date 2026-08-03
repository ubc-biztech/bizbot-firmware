"""Hold W/A/S/D to drive BizBot over USB serial or local Wi-Fi."""

import argparse
import socket
import sys
import threading
import time

import keyboard
import serial
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
        print(">", cmd)

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
            print(f"Connection lost while sending: {error}", file=sys.stderr)
            running.clear()
            return

        time.sleep(KEEPALIVE_INTERVAL_S)


def receive_loop(transport):
    while running.is_set():
        try:
            line = transport.read_line()
        except (OSError, serial.SerialException) as error:
            print(f"Connection lost while receiving: {error}", file=sys.stderr)
            running.clear()
            return

        if line:
            print("<", line)


def movement_command():
    """Return the command for currently held WASD keys, or STOP if none."""
    forward = keyboard.is_pressed("w")
    backward = keyboard.is_pressed("s")
    left = keyboard.is_pressed("a")
    right = keyboard.is_pressed("d")

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
    # Keep the robot stationary while we type. The keepalive thread keeps
    # sending STOP in the background, so the firmware link never times out.
    set_keepalive("STOP")

    # Drop the 't' keystrokes (and anything else) already sitting in the
    # terminal input buffer so the prompt starts empty.
    if sys.platform != "win32":
        import termios
        time.sleep(0.2)  # give the held key time to be released
        termios.tcflush(sys.stdin, termios.TCIFLUSH)

    print()
    raw = input("SET_PID kp ki kd (blank to cancel): ").strip()

    if not raw:
        print("Tuning cancelled.")
        return

    parts = raw.split()
    if len(parts) != 3:
        print("Need exactly three numbers, e.g.: 0.08 0 0.002")
        return

    try:
        kp, ki, kd = (float(part) for part in parts)
    except ValueError:
        print(f"Not numbers: {raw}")
        return

    send(transport, f"SET_PID {kp} {ki} {kd}")
    time.sleep(0.3)  # let the response print and the T key release


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

    try:
        send(transport, "ENABLE")
        set_keepalive("STOP")

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

        print(
            "Hold W/A/S/D to move. Release to stop. "
            "I = state. T = tune PID. Q = quit."
        )

        while running.is_set():
            if keyboard.is_pressed("q"):
                break

            if keyboard.is_pressed("t"):
                handle_tuning_input(transport)
            elif keyboard.is_pressed("i"):
                set_keepalive("STOP")
                send(transport, "GET_STATE")
                time.sleep(0.2)
            else:
                set_keepalive(movement_command())

            time.sleep(POLL_INTERVAL_S)

    finally:
        running.clear()

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
