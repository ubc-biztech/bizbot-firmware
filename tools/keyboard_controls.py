"""
Hold W/A/S/D to drive BizBot over USB serial. Release to stop. Q to quit.

Setup:
    pip install pyserial keyboard

macOS: System Settings → Privacy & Security → Accessibility → allow Terminal/Cursor.
Linux: may need sudo for global key capture.
"""

import sys
import threading
import time

import keyboard
import serial
from serial.tools import list_ports

BAUD = 115200
KEEPALIVE_INTERVAL_S = 0.2
POLL_INTERVAL_S = 0.05

LINEAR_SPEED = 0.25
ANGULAR_SPEED = 0.25

running = threading.Event()
running.set()

keepalive_cmd = "STOP"
keepalive_lock = threading.Lock()
serial_lock = threading.Lock()


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


def send(ser, cmd, echo=True):
    if echo:
        print(">", cmd)

    with serial_lock:
        ser.write((cmd + "\n").encode())


def set_keepalive(cmd):
    global keepalive_cmd

    with keepalive_lock:
        keepalive_cmd = cmd


def get_keepalive():
    with keepalive_lock:
        return keepalive_cmd


def keepalive_loop(ser):
    while running.is_set():
        cmd = get_keepalive()

        # Important: send repeatedly, not only when the command changes.
        send(ser, cmd, echo=False)

        time.sleep(KEEPALIVE_INTERVAL_S)


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


def main():
    port = find_serial_port()
    print(f"Opening {port} at {BAUD} baud...")

    ser = serial.Serial(port, BAUD, timeout=0.1)
    time.sleep(2)  # wait for ESP32 reset after USB connect

    try:
        send(ser, "ENABLE")
        set_keepalive("STOP")

        keepalive_thread = threading.Thread(
            target=keepalive_loop,
            args=(ser,),
        )
        keepalive_thread.start()

        print("Hold W/A/S/D to move. Release to stop. I = state. Q = quit.")

        while True:
            if keyboard.is_pressed("q"):
                break

            if keyboard.is_pressed("i"):
                send(ser, "GET_STATE")
                time.sleep(0.2)
            else:
                set_keepalive(movement_command())

            time.sleep(POLL_INTERVAL_S)

    finally:
        running.clear()
        keepalive_thread.join(timeout=1)

        try:
            send(ser, "STOP")
            send(ser, "DISABLE")
        finally:
            ser.close()
            print("Disconnected.")


if __name__ == "__main__":
    try:
        main()
    except serial.SerialException as error:
        print(f"Serial error: {error}", file=sys.stderr)
        print("Tip: check USB cable and set the correct port.", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)