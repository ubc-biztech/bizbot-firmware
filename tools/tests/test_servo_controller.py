"""Run using a Python with Tkinter: python3 -m unittest discover -s tools/tests."""
import importlib.util
from pathlib import Path
import queue
import socket
import threading
import unittest

spec = importlib.util.spec_from_file_location("servo_controller", Path(__file__).parents[1] / "servo_controller.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ServoTests(unittest.TestCase):
    def test_angles(self):
        self.assertEqual(module.servo_command(1, 0), "SERVO 1 0.0")
        self.assertEqual(module.servo_command(3, 180), "SERVO 3 180.0")
        for number, angle in [(0, 90), (4, 90), (1, -1), (1, 181), (1, float('nan')), (1, float('inf'))]:
            with self.assertRaises(ValueError):
                module.servo_command(number, angle)

    def test_fragmented_reply_and_telemetry(self):
        client, server = socket.socketpair()
        self.addCleanup(client.close)
        self.addCleanup(server.close)
        link = module.Connection("Wi-Fi", "unused", queue.Queue())
        link.transport = client
        client.settimeout(0.1)
        def respond():
            self.assertEqual(server.recv(100), b"SERVO 2 90.0\n")
            server.sendall(b"STATE pitch=0\nOK SER")
            server.sendall(b"VO 2 90.0\n")
        worker = threading.Thread(target=respond)
        worker.start()
        self.assertEqual(link.exchange("SERVO 2 90.0"), "OK SERVO 2 90.0")
        worker.join()

    def test_unknown_firmware_error(self):
        client, server = socket.socketpair()
        self.addCleanup(client.close)
        self.addCleanup(server.close)
        link = module.Connection("Wi-Fi", "unused", queue.Queue())
        link.transport = client
        server.sendall(b"ERR UNKNOWN_COMMAND\n")
        with self.assertRaisesRegex(ConnectionError, 'new firmware'):
            link.exchange("SERVO_PING")

    def test_disconnect_detected(self):
        client, server = socket.socketpair()
        self.addCleanup(client.close)
        self.addCleanup(server.close)
        link = module.Connection("Wi-Fi", "unused", queue.Queue())
        link.transport = client
        server.shutdown(socket.SHUT_WR)
        with self.assertRaises(ConnectionError):
            link.exchange("SERVO_PING")


if __name__ == '__main__':
    unittest.main()
