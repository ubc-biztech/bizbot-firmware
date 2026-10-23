import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from controls import ControlPanel
from drive import DriveController


class DriveTests(unittest.TestCase):
    def setUp(self):
        self.driver = DriveController(.01, 40, 20, 2)
        self.obs = np.zeros((1, 162), dtype=np.float32)
        self.radius = .0825

    def tick(self, direction):
        adapted = self.driver.adapt(self.obs, self.radius, direction)
        self.driver.record(self.obs, self.radius)
        return adapted

    def test_idle_is_exact_passthrough(self):
        self.obs[0, -1] = .3
        before = self.obs.copy()
        for _ in range(100):
            self.assertIs(self.tick(0), self.obs)
        np.testing.assert_array_equal(self.obs, before)

    def test_hold_reverse_and_release_limits(self):
        references = []
        for direction, steps in [(1, 500), (-1, 900), (0, 900)]:
            for _ in range(steps):
                self.tick(direction)
                references.append(self.driver.reference)
            if direction:
                self.assertAlmostEqual(self.driver.reference, direction * .1, places=3)
        self.assertLess(abs(self.driver.reference), 1e-5)
        self.assertLessEqual(max(abs(np.array(references))), .1)
        self.assertLessEqual(max(abs(np.diff(references))) / .01, .080001)

    def test_adapter_preserves_sensors_and_uses_reference_history(self):
        self.obs[0, -1] = .25
        before = self.obs.copy()
        self.tick(1)
        refs = self.driver.reference_history.copy()
        integral = self.driver.integral
        adapted = self.driver.adapt(self.obs, self.radius, 1)
        np.testing.assert_array_equal(self.obs, before)
        np.testing.assert_array_equal(adapted[0, :-2].reshape(40, 4)[:, [0, 1, 3]], 0)
        np.testing.assert_allclose(adapted[0, :-2].reshape(40, 4)[:, 2], -refs)
        self.assertAlmostEqual(float(adapted[0, -1]), integral / 2, places=6)

    def test_recovery_requires_release(self):
        for _ in range(300):
            self.tick(1)
        self.obs[0, :-2].reshape(40, 4)[:, 0] = 1.4
        self.tick(1)
        self.assertTrue(self.driver.require_release)
        self.obs[:] = 0
        for _ in range(500):
            self.tick(1)
        self.assertTrue(self.driver.require_release)
        self.assertLess(self.driver.reference, .001)
        self.tick(0)
        self.assertFalse(self.driver.require_release)
        for _ in range(100):
            self.tick(1)
        self.assertGreater(self.driver.reference, 0)

    def test_reset_clears_drive_and_waits_for_release(self):
        for _ in range(100):
            self.tick(1)
        self.driver.reset(require_release=True)
        self.assertEqual(self.driver.reference, 0)
        np.testing.assert_array_equal(self.driver.reference_history, 0)
        self.assertIs(self.tick(1), self.obs)
        self.tick(0)
        self.tick(1)
        self.assertGreater(self.driver.reference, 0)

    def test_stale_input_cannot_restart_after_release(self):
        import queue
        panel = ControlPanel(queue.Queue())
        try:
            with patch('controls.time.monotonic', return_value=10):
                panel.receive_drive(1, 10)
                self.assertEqual(panel.direction(), 1)
                panel.receive_drive(0, 12)
                panel.receive_drive(1, 11)
                self.assertEqual(panel.direction(), 0)
                panel.receive_drive(-1, 13)
            with patch('controls.time.monotonic', return_value=10.36):
                self.assertEqual(panel.direction(), 0)
        finally:
            panel.server.server_close()

    def test_invalid_drive_packets(self):
        for packet in [dict(direction=2, sequence=1), dict(direction=1, sequence=-1),
                       dict(direction=True, sequence=1), dict(direction=1, sequence=1.5)]:
            with self.assertRaises(ValueError):
                ControlPanel.command(dict(name='drive', **packet))

    def test_live_limit_changes_are_smooth_and_survive_reset(self):
        for direction in (1, -1):
            self.driver.reset()
            self.driver.set_speed_limit(.25)
            references = []
            for limit in (.25, .04, .20, 0):
                self.driver.set_speed_limit(limit)
                for _ in range(800):
                    self.tick(direction)
                    references.append(self.driver.reference)
                self.assertAlmostEqual(self.driver.reference, direction * limit, places=3)
            self.assertLessEqual(max(abs(np.diff(references))) / .01, .080001)
        self.driver.set_speed_limit(.23)
        self.driver.reset(require_release=True)
        self.assertEqual(self.driver.state()['limit'], .23)

    def test_speed_limit_validation(self):
        self.driver.set_speed_limit(2.0)
        self.assertEqual(self.driver.state()['limit'], 2.0)
        self.assertEqual(ControlPanel.command(dict(name='speed_limit', value=2.0)), 'speed_limit 2.0')
        self.assertEqual(ControlPanel.command(dict(name='speed_limit', value=.2)), 'speed_limit 0.2')
        for value in (-.01, 10.01, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.driver.set_speed_limit(value)
            with self.assertRaises(ValueError):
                ControlPanel.command(dict(name='speed_limit', value=value))

    def test_acceleration_limit_changes_ramp_and_survives_reset(self):
        self.driver.set_speed_limit(2.0)
        for _ in range(100):
            self.tick(1)
        slow_reference = self.driver.reference
        self.driver.reset()
        self.driver.set_acceleration_limit(10.0)
        for _ in range(100):
            self.tick(1)
        self.assertGreater(self.driver.reference, slow_reference)
        self.driver.reset()
        self.assertEqual(self.driver.state()['acceleration_limit'], 10.0)
        self.assertEqual(self.driver.limits.braking, .08)

    def test_acceleration_limit_validation(self):
        self.assertEqual(ControlPanel.command(dict(name='acceleration_limit', value=10.0)), 'acceleration_limit 10.0')
        for value in (0, -.01, 50.01, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.driver.set_acceleration_limit(value)
            with self.assertRaises(ValueError):
                ControlPanel.command(dict(name='acceleration_limit', value=value))

    def test_expanded_motion_ranges(self):
        self.driver.set_speed_limit(10)
        self.driver.set_acceleration_limit(50)
        self.assertEqual(self.driver.state()['limit'], 10)
        self.assertEqual(self.driver.state()['acceleration_limit'], 50)
        self.assertEqual(ControlPanel.command(dict(name='speed_limit', value=10)), 'speed_limit 10.0')
        self.assertEqual(ControlPanel.command(dict(name='acceleration_limit', value=50)), 'acceleration_limit 50.0')

    def test_selected_acceleration_controls_start_and_stop_without_overshoot(self):
        for direction in (1, -1):
            for acceleration in (.1, 1, 10):
                self.driver.reset()
                self.driver.set_speed_limit(.3)
                self.driver.set_acceleration_limit(acceleration)
                steps = int(np.ceil(.3 / acceleration / self.driver.dt))
                for _ in range(steps):
                    before = self.driver.reference
                    self.tick(direction)
                    self.assertLessEqual(abs(self.driver.reference), .3)
                    self.assertLessEqual(abs(self.driver.reference - before), acceleration * .01 + 1e-12)
                self.assertAlmostEqual(self.driver.reference, direction * .3)
                before = self.driver.reference
                self.tick(0)
                self.assertLess(abs(self.driver.reference), abs(before))
                for _ in range(steps):
                    self.tick(0)
                    self.assertGreaterEqual(direction * self.driver.reference, 0)
                self.assertEqual(self.driver.reference, 0)

    def test_release_during_acceleration_brakes_on_next_step(self):
        self.driver.set_speed_limit(2)
        self.driver.set_acceleration_limit(10)
        self.tick(1)
        self.assertGreater(self.driver.acceleration, 0)
        self.tick(0)
        self.assertLess(self.driver.acceleration, 0)
        self.assertEqual(self.driver.reference, 0)

    def test_request_stays_bounded_at_high_acceleration(self):
        self.driver.set_speed_limit(2)
        self.driver.set_acceleration_limit(10)
        for _ in range(1000):
            self.tick(1)
            self.assertLessEqual(self.driver.reference, 2)
        self.assertEqual(self.driver.reference, 2)

    def test_speed_limit_holds_motion_without_rearm(self):
        for direction in (1, -1):
            self.driver.reset()
            self.obs[:] = 0
            self.driver.set_speed_limit(.3)
            self.driver.set_acceleration_limit(1)
            for _ in range(40):
                self.tick(direction)
            frame = self.obs[0, :-2].reshape(40, 4)
            frame[:, 2] = direction * .5 / (2 * np.pi * self.radius)
            self.tick(direction)
            self.assertFalse(self.driver.require_release)
            self.assertEqual(self.driver.reference, direction * .3)
            self.assertEqual(self.driver.acceleration, 0)
            self.assertEqual(self.driver.status, 'Driving')
            self.tick(0)
            self.assertLess(abs(self.driver.reference), .3)

    def test_measured_limit_pauses_acceleration_and_resumes_below_limit(self):
        for direction in (1, -1):
            self.driver.reset()
            self.obs[:] = 0
            self.driver.set_speed_limit(.3)
            self.driver.set_acceleration_limit(1)
            for _ in range(10):
                self.tick(direction)
            before = self.driver.reference
            frame = self.obs[0, :-2].reshape(40, 4)
            frame[:, 2] = direction * .4 / (2 * np.pi * self.radius)
            self.tick(direction)
            self.assertEqual(self.driver.reference, before)
            self.assertEqual(self.driver.acceleration, 0)
            self.assertFalse(self.driver.require_release)
            frame[:, 2] = 0
            self.tick(direction)
            self.assertGreater(abs(self.driver.reference), abs(before))


if __name__ == '__main__':
    unittest.main()
