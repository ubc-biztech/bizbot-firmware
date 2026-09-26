"""Test the actual tuning prompt without loading desktop keyboard/serial drivers."""
import ast
import math
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class VelocityTuningTest(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[2] / "tools" / "keyboard_controls.py"
        tree = ast.parse(source.read_text())
        prompt = next(node for node in tree.body
                      if isinstance(node, ast.FunctionDef) and node.name == "handle_tuning_input")
        self.sent = Mock()
        self.keepalive = Mock()
        running = threading.Event()
        running.set()
        scope = dict(math=math, sys=SimpleNamespace(platform="win32"),
                     time=SimpleNamespace(sleep=Mock()), running=running,
                     send=self.sent, set_keepalive=self.keepalive)
        exec(compile(ast.Module(body=[prompt], type_ignores=[]), str(source), "exec"), scope)
        self.prompt = scope["handle_tuning_input"]

    def enter(self, value):
        with patch("builtins.input", return_value=value), patch("builtins.print"):
            self.prompt("transport")
        self.keepalive.assert_called_with("STOP")

    def test_live_velocity_gain(self):
        self.enter("vel 0.005")
        self.sent.assert_called_once_with("transport", "SET_VEL_KP 0.005")

    def test_zero_disables_correction(self):
        self.enter("vel 0")
        self.sent.assert_called_once_with("transport", "SET_VEL_KP 0.0")

    def test_invalid_values_send_nothing(self):
        for value in ("vel", "vel 1 2", "vel nope", "vel nan", "vel inf", "vel -1", ""):
            with self.subTest(value=value):
                self.enter(value)
                self.sent.assert_not_called()

    def test_angle_tuning_is_preserved(self):
        self.enter("0.04 0 0.001")
        self.sent.assert_called_once_with("transport", "SET_PID 0.04 0.0 0.001")


if __name__ == "__main__":
    unittest.main()
