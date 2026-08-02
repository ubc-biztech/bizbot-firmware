"""
Live visualization of BizBot's pitch from the IMU serial debug stream.

Left: side view of the two-wheeled robot leaning by the reported pitch.
Right: rolling pitch and pitch-rate charts.

Reads the lines printed by the firmware when built with -DIMU_DEBUG_EULER=1:
    IMU rpy(deg)=1.2,3.4,5.6 pitch=3.4 rate(dps)=0.7

Setup:
    pip install pyserial matplotlib

Usage:
    python tools/imu_visualizer.py            # auto-detect serial port
    python tools/imu_visualizer.py --port /dev/cu.usbserial-0001
    python tools/imu_visualizer.py --demo     # no hardware; synthetic wobble
"""

import argparse
import collections
import math
import re
import sys
import threading
import time

import matplotlib.pyplot as plt
from matplotlib import animation, patches, transforms

BAUD = 115200
WINDOW_S = 10.0          # rolling chart window
MAX_TILT_DEG = 35.0      # firmware cutoff (config.h MAX_TILT_DEG)
STALE_S = 0.5            # no packet for this long => "no data" banner

# Colors (validated reference palette, light mode)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES_PITCH = "#2a78d6"   # blue
SERIES_RATE = "#eb6834"    # orange
STATUS_CRITICAL = "#d03b3b"

# Firmware debug line, e.g. "IMU rpy(deg)=1.2,3.4,5.6 pitch=3.4 rate(dps)=0.7"
LINE_RE = re.compile(
    r"IMU rpy\(deg\)="
    r"(?P<roll>-?\d+\.?\d*),(?P<pitch_raw>-?\d+\.?\d*),(?P<yaw>-?\d+\.?\d*)"
    r"\s+pitch=(?P<pitch>-?\d+\.?\d*)"
    r"\s+rate\(dps\)=(?P<rate>-?\d+\.?\d*)"
)


def parse_line(line):
    """Return (pitch_deg, rate_dps) from a firmware debug line, or None."""
    match = LINE_RE.search(line)
    if not match:
        return None
    return float(match.group("pitch")), float(match.group("rate"))


def find_serial_port():
    """Pick the first likely USB serial device (same heuristic as keyboard_controls)."""
    from serial.tools import list_ports

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


class Telemetry:
    """Thread-safe rolling buffer of (t, pitch, rate) samples."""

    def __init__(self):
        self.lock = threading.Lock()
        self.samples = collections.deque(maxlen=4096)
        self.start = time.monotonic()

    def add(self, pitch, rate):
        now = time.monotonic() - self.start
        with self.lock:
            self.samples.append((now, pitch, rate))

    def snapshot(self):
        now = time.monotonic() - self.start
        with self.lock:
            recent = [s for s in self.samples if now - s[0] <= WINDOW_S]
        return now, recent


def serial_reader(telemetry, port, stop):
    import serial

    print(f"Opening {port} at {BAUD} baud...")
    ser = serial.Serial(port, BAUD, timeout=0.2)
    try:
        while not stop.is_set():
            line = ser.readline().decode(errors="replace")
            parsed = parse_line(line)
            if parsed:
                telemetry.add(*parsed)
    finally:
        ser.close()


def demo_reader(telemetry, stop):
    """Synthetic damped wobble + noise, so the display works with no hardware."""
    import random

    t0 = time.monotonic()
    while not stop.is_set():
        t = time.monotonic() - t0
        cycle = t % 12.0
        # Re-kick a decaying oscillation every 12 s, with a large excursion
        # near the end of the cycle to exercise the tilt-limit state.
        amp = 25.0 if cycle > 10.0 else 8.0 * math.exp(-0.15 * cycle)
        pitch = amp * math.sin(2.0 * math.pi * 0.8 * t) + random.gauss(0, 0.3)
        rate = amp * 2.0 * math.pi * 0.8 * math.cos(2.0 * math.pi * 0.8 * t)
        telemetry.add(pitch, rate)
        time.sleep(0.01)  # ~100 Hz, like the RVC stream


def build_robot(ax):
    """Draw the side-view robot; return the artists that move with pitch."""
    ax.set_facecolor(SURFACE)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-0.55, 2.3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    wheel_r = 0.42

    # Ground
    ax.axhline(-wheel_r, color=BASELINE, linewidth=2, zorder=1)
    for x in [i * 0.25 - 1.5 for i in range(13)]:
        ax.plot(
            [x, x - 0.12], [-wheel_r, -wheel_r - 0.12],
            color=GRID, linewidth=1.5, zorder=1,
        )

    # Wheel, fixed at the origin (axle)
    ax.add_patch(
        patches.Circle((0, 0), wheel_r, fill=False, color=INK, linewidth=2, zorder=3)
    )
    ax.add_patch(patches.Circle((0, 0), 0.05, color=INK, zorder=4))

    # Body: a rounded rectangle standing up from the axle; rotated by pitch.
    body_w, body_h = 0.5, 1.55
    body = patches.FancyBboxPatch(
        (-body_w / 2, wheel_r * 0.5),
        body_w,
        body_h,
        boxstyle="round,pad=0.04,rounding_size=0.1",
        facecolor=SERIES_PITCH,
        edgecolor="none",
        zorder=2,
    )
    ax.add_patch(body)

    # Head marker on top of the body so direction of lean is obvious.
    head = patches.Circle((0, wheel_r * 0.5 + body_h + 0.12), 0.1,
                          color=SERIES_PITCH, zorder=2)
    ax.add_patch(head)

    # Upright reference through the axle.
    ax.plot([0, 0], [0, 2.2], color=GRID, linewidth=1, linestyle=(0, (4, 4)), zorder=1)

    pitch_text = ax.text(
        0.03, 0.97, "", transform=ax.transAxes, ha="left", va="top",
        fontsize=13, color=INK, family="sans-serif",
    )
    status_text = ax.text(
        0.03, 0.89, "", transform=ax.transAxes, ha="left", va="top",
        fontsize=11, color=STATUS_CRITICAL, family="sans-serif",
    )
    return body, head, pitch_text, status_text


def style_chart(ax, ylabel):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(BASELINE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=10)
    ax.axhline(0, color=BASELINE, linewidth=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--port", help="serial port (default: auto-detect)")
    parser.add_argument("--demo", action="store_true",
                        help="synthetic data, no hardware needed")
    parser.add_argument("--save-frame", metavar="PATH",
                        help="render one frame to an image and exit (headless check)")
    args = parser.parse_args()

    telemetry = Telemetry()
    stop = threading.Event()

    if args.demo:
        reader = threading.Thread(target=demo_reader, args=(telemetry, stop))
    else:
        port = args.port or find_serial_port()
        reader = threading.Thread(target=serial_reader, args=(telemetry, port, stop))
    reader.daemon = True
    reader.start()

    fig = plt.figure(figsize=(11, 5.5), facecolor=SURFACE)
    fig.canvas.manager.set_window_title("BizBot IMU visualizer")
    grid_spec = fig.add_gridspec(2, 2, width_ratios=[1, 1.4], hspace=0.15, wspace=0.12)

    ax_robot = fig.add_subplot(grid_spec[:, 0])
    ax_pitch = fig.add_subplot(grid_spec[0, 1])
    ax_rate = fig.add_subplot(grid_spec[1, 1], sharex=ax_pitch)

    body, head, pitch_text, status_text = build_robot(ax_robot)

    # One series per axis: pitch and rate have different units, so they get
    # separate charts rather than a second y-axis.
    style_chart(ax_pitch, "pitch (deg)")
    style_chart(ax_rate, "rate (deg/s)")
    plt.setp(ax_pitch.get_xticklabels(), visible=False)
    ax_rate.set_xlabel("time (s)", color=INK_MUTED, fontsize=10)

    ax_pitch.axhline(MAX_TILT_DEG, color=STATUS_CRITICAL, linewidth=1,
                     linestyle=(0, (4, 4)))
    ax_pitch.axhline(-MAX_TILT_DEG, color=STATUS_CRITICAL, linewidth=1,
                     linestyle=(0, (4, 4)))
    ax_pitch.text(0.995, MAX_TILT_DEG, "tilt cutoff ", ha="right", va="bottom",
                  transform=ax_pitch.get_yaxis_transform(),
                  fontsize=8, color=STATUS_CRITICAL)

    (pitch_line,) = ax_pitch.plot([], [], color=SERIES_PITCH, linewidth=2)
    (rate_line,) = ax_rate.plot([], [], color=SERIES_RATE, linewidth=2)

    def update(_frame):
        now, samples = telemetry.snapshot()

        if not samples or now - samples[-1][0] > STALE_S:
            status_text.set_text("⚠ NO DATA")
            body.set_facecolor(INK_MUTED)
            head.set_color(INK_MUTED)
            return

        pitch = samples[-1][1]

        rot = transforms.Affine2D().rotate_deg(-pitch) + ax_robot.transData
        body.set_transform(rot)
        head.set_transform(rot)

        pitch_text.set_text(f"pitch {pitch:+.1f}°")
        if abs(pitch) > MAX_TILT_DEG:
            status_text.set_text("⚠ TILT LIMIT")
            body.set_facecolor(STATUS_CRITICAL)
            head.set_color(STATUS_CRITICAL)
        else:
            status_text.set_text("")
            body.set_facecolor(SERIES_PITCH)
            head.set_color(SERIES_PITCH)

        times = [s[0] for s in samples]
        pitch_line.set_data(times, [s[1] for s in samples])
        rate_line.set_data(times, [s[2] for s in samples])

        ax_pitch.set_xlim(max(0.0, now - WINDOW_S), max(now, WINDOW_S))
        pitch_span = max(40.0, max(abs(s[1]) for s in samples) * 1.2)
        rate_span = max(60.0, max(abs(s[2]) for s in samples) * 1.2)
        ax_pitch.set_ylim(-pitch_span, pitch_span)
        ax_rate.set_ylim(-rate_span, rate_span)

    if args.save_frame:
        time.sleep(0.5)  # let the reader collect some samples first
        update(0)
        fig.savefig(args.save_frame, dpi=110)
        stop.set()
        reader.join(timeout=1)
        print(f"Saved {args.save_frame}")
        return

    anim = animation.FuncAnimation(fig, update, interval=33, cache_frame_data=False)

    try:
        plt.show()
    finally:
        stop.set()
        reader.join(timeout=1)

    del anim


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
