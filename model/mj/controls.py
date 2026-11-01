import json
import math
import re
import struct
import threading
import time
import webbrowser
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

PUSH_DIRECTIONS = ('forward', 'back', 'left', 'right')   # relative to where the robot faces
MODE_NAME = re.compile(r'[A-Za-z0-9 _.\-]{1,32}')   # PID, a policy label or an LQR; watch.py checks it is one of its tabs
PID_LIMITS = ((0.0, 1.0), (0.0, 10.0), (0.0, 0.1))        # kp, ki, kd


def png(rgb):
    """Encode an (h, w, 3) uint8 image as PNG (fast zlib level, no filtering)."""
    h, w, _ = rgb.shape
    raw = np.zeros((h, w * 3 + 1), np.uint8)   # each row starts with filter type 0
    raw[:, 1:] = rgb.reshape(h, w * 3)
    chunk = lambda kind, body: (struct.pack('>I', len(body)) + kind + body
                                + struct.pack('>I', zlib.crc32(kind + body) & 0xffffffff))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(raw.tobytes(), 1)) + chunk(b'IEND', b''))


class ControlPanel:
    def __init__(self, commands):
        self.commands = commands
        self.state = {}
        self.lock = threading.Lock()
        self.frame = None              # latest rendered view, (h, w, 3) uint8
        self.frame_seq = 0             # number of the latest view
        self.frame_ready = threading.Condition()
        self.frame_png = (0, b'')      # (view number it was encoded from, PNG bytes)
        self.frame_requested = None    # monotonic time the page last asked for the view
        self.drive_direction = 0
        self.turn_direction = 0
        self.drive_updated = 0.0
        self.drive_sequence = -1
        panel = self
        page = Path(__file__).with_name('controls.html').read_bytes()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, status, body, content_type='application/json'):
                self.send_response(status)
                self.send_header('Content-Type', content_type)
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                url = urlsplit(self.path)
                if self.path == '/':
                    self.reply(200, page, 'text/html; charset=utf-8')
                elif url.path == '/frame':
                    # Answers as soon as there is a view newer than ?after=<number> (or after 1 s).
                    panel.frame_requested = time.monotonic()
                    try:
                        after = int(parse_qs(url.query).get('after', ['0'])[0])
                    except ValueError:
                        after = 0
                    with panel.frame_ready:
                        panel.frame_ready.wait_for(lambda: panel.frame_seq > after, timeout=1.0)
                        seq, frame = panel.frame_seq, panel.frame
                    if frame is None:
                        self.reply(204, b'', 'image/png')
                        return
                    if panel.frame_png[0] != seq:
                        panel.frame_png = (seq, png(frame))
                    self.send_response(200)
                    self.send_header('Content-Type', 'image/png')
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('X-Frame', str(panel.frame_png[0]))
                    self.send_header('Content-Length', str(len(panel.frame_png[1])))
                    self.end_headers()
                    self.wfile.write(panel.frame_png[1])
                elif self.path == '/state':
                    with panel.lock:
                        body = json.dumps(panel.state).encode()
                    self.reply(200, body)
                else:
                    self.reply(404, b'{}')

            def do_POST(self):
                if self.path != '/command':
                    self.reply(404, b'{}')
                    return
                if self.headers.get('Origin') not in (None, panel.url):
                    self.reply(403, b'{}')
                    return
                try:
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 0 < size <= 1024:
                        raise ValueError('Invalid request size')
                    command = panel.command(json.loads(self.rfile.read(size)))
                except (ValueError, TypeError, KeyError, AttributeError):
                    self.reply(400, b'{"error":"Invalid control value"}')
                    return
                if command.startswith('drive '):
                    _, direction, sequence, turn = command.split()
                    panel.receive_drive(int(direction), int(sequence), int(turn))
                else:
                    panel.commands.put(command)
                self.reply(202, b'{}')

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.url = f'http://127.0.0.1:{self.server.server_port}'
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @staticmethod
    def command(data):
        name = data['name']
        if name == 'new':
            return 'new'
        if name == 'drive':
            direction, sequence = data['direction'], data['sequence']
            turn = data.get('turn', 0)
            if type(direction) is not int or direction not in (-1, 0, 1):
                raise ValueError('Invalid direction')
            if type(sequence) is not int or not 0 <= sequence < 2**53:
                raise ValueError('Invalid sequence')
            if type(turn) is not int or turn not in (-1, 0, 1):
                raise ValueError('Invalid turn')
            return f'drive {direction} {sequence} {turn}'
        if name == 'mode':
            if type(data['value']) is not str or not MODE_NAME.fullmatch(data['value']):
                raise ValueError('Invalid mode')
            return f"mode {data['value']}"
        if name == 'pid':
            gains = [float(data[k]) for k in ('kp', 'ki', 'kd')]
            if not all(math.isfinite(g) and lo <= g <= hi for g, (lo, hi) in zip(gains, PID_LIMITS)):
                raise ValueError('Invalid gains')
            return 'pid ' + ' '.join(repr(g) for g in gains)
        if name == 'cam':
            azimuth, elevation, zoom = float(data['azimuth']), float(data['elevation']), float(data['zoom'])
            if not all(map(math.isfinite, (azimuth, elevation, zoom))) or abs(azimuth) > 360 or abs(elevation) > 180 or abs(zoom) > 2:
                raise ValueError('Invalid camera move')
            return f'cam {azimuth} {elevation} {zoom}'
        if name == 'push':
            force, duration, direction = float(data['force']), float(data['duration']), data['direction']
            if not math.isfinite(force) or not 1 <= force <= 150 or not .05 <= duration <= 2 or direction not in PUSH_DIRECTIONS:
                raise ValueError('Invalid push')
            return f'push {force} {duration} {direction}'
        limits = {'com': (-15, 15), 'tilt': (-20, 20), 'lat': (0, 250),
                  'speed_limit': (0, 10.0), 'acceleration_limit': (.01, 50.0), 'turn_limit': (0, 720),
                  'turn_acceleration': (1, 3600), 'turn_command_limit': (0, 100)}
        lo, hi = limits[name]
        value = float(data['value'])
        if not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError('Invalid value')
        return f'{name} {value}'

    def receive_drive(self, direction, sequence, turn=0):
        with self.lock:
            if sequence > self.drive_sequence:
                self.drive_direction = direction
                self.turn_direction = turn
                self.drive_sequence = sequence
                self.drive_updated = time.monotonic()

    def direction(self):
        return self.inputs()[0]

    def inputs(self):
        with self.lock:
            return (self.drive_direction, self.turn_direction) if time.monotonic() - self.drive_updated <= .35 else (0, 0)

    def show(self, frame):
        """Hand the page a new view of the scene."""
        with self.frame_ready:
            self.frame = frame
            self.frame_seq += 1
            self.frame_ready.notify_all()

    def page_closed(self, timeout=5.0):
        """True once the page has shown the view and then stopped asking for it (the tab was closed)."""
        return self.frame_requested is not None and time.monotonic() - self.frame_requested > timeout

    def publish(self, plant, drive=None, turn=None, robot='', push=None, extra=None):
        p = plant.p
        with self.lock:
            self.state = {'robot': robot,
                          'push_key': push,
                          'com': float(p['com_offset'][0] * 100),
                          'tilt': float(p['imu_mount_deg'][0]),
                          'lat': float(p['imu_latency'][0] * 1000),
                          'push': float(plant.push_F[0]) if plant.t[0] < plant.push_until[0] else 0.0,   # random pushes (--pushes)
                          'drive': drive.state() if drive is not None else None,
                          'turn': turn.state() if turn is not None else None,
                          **(extra or {})}

    def __enter__(self):
        self.thread.start()
        print(f'Controls: {self.url}', flush=True)
        webbrowser.open(self.url)
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)
