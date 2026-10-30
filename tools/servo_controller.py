#!/usr/bin/env python3
"""Three-servo GUI. Run: python3 tools/servo_controller.py
USB requires pyserial; Wi-Fi uses only Python's standard library.
"""
import queue
import socket
import threading
import time
try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    raise SystemExit("Tkinter is missing. For Homebrew Python 3.13, run: brew install python-tk@3.13\n"
                     "Then restart this script with python3.")

PINS = (18, 19, 23)


def servo_command(number, angle):
    angle = float(angle)
    if number not in (1, 2, 3) or not 0 <= angle <= 180:
        raise ValueError("Choose a servo from 1–3 and an angle from 0–180.")
    return f"SERVO {number} {angle:.1f}"


class Connection(threading.Thread):
    def __init__(self, mode, address, events):
        super().__init__(daemon=True)
        self.mode, self.address, self.events = mode, address, events
        self.commands = queue.Queue(maxsize=10)
        self.stop_requested = threading.Event()
        self.transport = None
        self.buffer = b""

    def write(self, command):
        data = (command + "\n").encode("ascii")
        if self.mode == "Wi-Fi":
            self.transport.sendall(data)
        else:
            self.transport.write(data)

    def exchange(self, command):
        self.write(command)
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            if b"\n" not in self.buffer:
                try:
                    chunk = (self.transport.recv(4096) if self.mode == "Wi-Fi"
                             else self.transport.read(256))
                except socket.timeout:
                    continue
                if not chunk and self.mode == "Wi-Fi":
                    raise ConnectionError("ESP32 disconnected")
                self.buffer += chunk
                if len(self.buffer) > 16384:
                    raise ConnectionError("Unexpected oversized response")
                if b"\n" not in self.buffer:
                    continue
            line, self.buffer = self.buffer.split(b"\n", 1)
            reply = line.decode(errors="replace").strip()
            if reply.startswith("ERR"):
                raise ConnectionError(reply + " — check that the new firmware is uploaded")
            expected = "OK " + command.split()[0]
            if reply == expected or reply.startswith(expected + " "):
                return reply
        raise TimeoutError("No servo acknowledgement. Upload the new firmware; close other controllers.")

    def run(self):
        try:
            if self.mode == "Wi-Fi":
                host, separator, port = self.address.partition(":")
                self.transport = socket.create_connection((host, int(port) if separator else 3333), timeout=3)
                self.transport.settimeout(0.1)
            else:
                import serial
                self.transport = serial.Serial(self.address, 115200, timeout=0.1, write_timeout=0.5)
                # USB opening can reset the ESP32. Wait without blocking the GUI.
                if self.stop_requested.wait(2):
                    return
                self.transport.reset_input_buffer()
            self.exchange("SERVO_OFF")
            self.events.put(("connected", "Connected · servos released"))
            while not self.stop_requested.is_set():
                try:
                    command = self.commands.get(timeout=0.4)
                except queue.Empty:
                    command = "SERVO_PING"
                if self.stop_requested.is_set():
                    break
                reply = self.exchange(command)
                if command != "SERVO_PING":
                    self.events.put(("reply", reply))
        except Exception as exc:
            self.events.put(("error", str(exc)))
        finally:
            if self.transport is not None:
                try:
                    self.write("SERVO_OFF")
                except Exception:
                    pass
                self.transport.close()
            self.events.put(("closed", "Disconnected · pulses stop within 2 seconds"))


class Controller:
    def __init__(self, root):
        self.root = root
        self.link = None
        self.events = queue.Queue()
        self.ready = False
        self.closing = False
        self.last_error = None
        root.title("BizBot · Servo controller")
        root.minsize(660, 430)
        frame = ttk.Frame(root, padding=24)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Servo controller", font=("Helvetica", 22, "bold")).pack(anchor="w")
        ttk.Label(frame, text="Choose an angle, then press Move. Connecting does not move the servos.").pack(anchor="w", pady=(4, 18))
        connection = ttk.Frame(frame)
        connection.pack(fill="x")
        self.mode = tk.StringVar(value="USB")
        selector = ttk.Combobox(connection, textvariable=self.mode, values=("USB", "Wi-Fi"), width=8, state="readonly")
        selector.pack(side="left")
        selector.bind("<<ComboboxSelected>>", self.change_mode)
        self.address = tk.StringVar()
        self.address_box = ttk.Combobox(connection, textvariable=self.address, width=32)
        self.address_box.pack(side="left", padx=8, fill="x", expand=True)
        ttk.Button(connection, text="Refresh USB", command=self.refresh_ports).pack(side="left")
        self.connect_button = ttk.Button(connection, text="Connect", command=self.connect)
        self.connect_button.pack(side="left", padx=(8, 0))
        self.controls = []
        for number, pin in enumerate(PINS, 1):
            row = ttk.LabelFrame(frame, text=f"Servo {number} · GPIO{pin}", padding=12)
            row.pack(fill="x", pady=(14, 0))
            angle = tk.DoubleVar(value=90)
            scale = tk.Scale(row, from_=0, to=180, orient="horizontal", variable=angle,
                             resolution=1, length=310, highlightthickness=0)
            scale.pack(side="left", fill="x", expand=True)
            move = ttk.Button(row, text="Move", command=lambda n=number, a=angle: self.send(servo_command(n, a.get())))
            move.pack(side="left", padx=12)
            release = ttk.Button(row, text="Release", command=lambda n=number: self.send(f"SERVO_OFF {n}"))
            release.pack(side="left")
            self.controls.extend((move, release))
        release_all = ttk.Button(frame, text="Release all servos", command=lambda: self.send("SERVO_OFF"))
        release_all.pack(anchor="w", pady=(16, 8))
        self.controls.append(release_all)
        ttk.Label(frame, text="External servo power + shared GND. Release stops pulses; it does not cut power.\n"
                  "0–180° maps to 1000–2000 µs; actual travel depends on your servo.\n"
                  "Use positional servos. This controller does not enable the drive wheels.").pack(anchor="w")
        self.status = tk.StringVar(value="Disconnected")
        ttk.Label(frame, textvariable=self.status, wraplength=610).pack(anchor="w", pady=(12, 0))
        self.refresh_ports()
        self.set_ready(False)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(50, self.poll)

    def set_ready(self, ready):
        self.ready = ready
        for widget in self.controls:
            widget.configure(state="normal" if ready else "disabled")

    def refresh_ports(self):
        try:
            from serial.tools import list_ports
            ports = [port.device for port in list_ports.comports()]
        except ImportError:
            ports = []
            self.status.set("USB needs pyserial: python3 -m pip install pyserial")
        self.address_box.configure(values=ports)
        if self.mode.get() == "USB" and ports and not self.address.get():
            self.address.set(ports[0])

    def change_mode(self, _event=None):
        if self.mode.get() == "Wi-Fi":
            self.address.set("192.168.4.1:3333")
        else:
            self.address.set("")
            self.refresh_ports()

    def connect(self):
        if self.link is not None:
            self.set_ready(False)
            self.link.stop_requested.set()
            self.connect_button.configure(state="disabled")
            self.status.set("Disconnecting…")
            return
        if not self.address.get().strip():
            self.status.set("Select a USB port or enter a Wi-Fi address.")
            return
        self.last_error = None
        self.link = Connection(self.mode.get(), self.address.get().strip(), self.events)
        self.link.start()
        self.connect_button.configure(text="Disconnect")
        self.status.set("Connecting…")

    def send(self, command):
        if not self.ready or self.link is None:
            return
        try:
            self.link.commands.put_nowait(command)
            self.status.set("Sending: " + command)
        except queue.Full:
            self.status.set("Waiting for ESP32; try again shortly.")

    def poll(self):
        while True:
            try:
                kind, message = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "connected":
                if self.link and not self.link.stop_requested.is_set():
                    self.set_ready(True)
            elif kind == "error":
                self.last_error = message
                self.set_ready(False)
            elif kind == "closed":
                self.link = None
                self.set_ready(False)
                self.connect_button.configure(text="Connect", state="normal")
                message = self.last_error or message
            self.status.set(message)
        if self.closing and self.link is None:
            self.root.destroy()
            return
        self.root.after(50, self.poll)

    def close(self):
        self.closing = True
        self.set_ready(False)
        if self.link:
            self.link.stop_requested.set()
        else:
            self.root.destroy()


if __name__ == "__main__":
    app = tk.Tk()
    Controller(app)
    app.mainloop()
