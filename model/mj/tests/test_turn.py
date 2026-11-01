import queue
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'model/mj'), str(ROOT / 'sim')]
from bizbot_mj import Randomization, U
from bizbot_mj.plant import Plant
from controls import ControlPanel
from drive import DriveController
from turn import TurnController
from turning_plant import TurningPlant


class TurnTests(unittest.TestCase):
    def drive(self, **changes):
        values = dict(measured_speed=0, reference=0, require_release=False)
        values.update(changes)
        return SimpleNamespace(**values)

    def command(self, controller, request, drive=None, left=0, right=0):
        return controller.update(request, drive or self.drive(), left, right, .0825, .48, 1000, 500)

    def test_direction_rate_limit_and_release(self):
        for direction in (1, -1):
            turn = TurnController(.01)
            for _ in range(100):
                command = self.command(turn, direction)
            self.assertEqual(turn.reference, direction * .25)
            self.assertGreater(direction * command, 0)
            for _ in range(60):
                self.command(turn, 0)
            self.assertEqual(turn.reference, 0)
            self.assertEqual(self.command(turn, 0), 0)
            self.assertFalse(turn.engaged)

    def test_turns_while_driving_forward_or_backward(self):
        for speed in (.2, -.2):
            for direction in (1, -1):
                turn = TurnController(.01)
                drive = self.drive(measured_speed=speed, reference=speed)
                command = self.command(turn, direction, drive)
                self.assertGreater(direction * command, 0)
                self.assertGreater(direction * turn.reference, 0)
                self.assertEqual(drive.reference, speed)

    def test_live_turn_limit_and_reset(self):
        turn = TurnController(.01)
        turn.set_rate_limit(90)
        previous = 0.0
        for _ in range(400):
            self.command(turn, 1)
            self.assertLessEqual(abs(turn.reference - previous), .005001)
            previous = turn.reference
        self.assertAlmostEqual(turn.reference, np.pi/2)
        turn.set_rate_limit(30)
        for _ in range(250):
            self.command(turn, 1)
        self.assertAlmostEqual(turn.reference, np.pi/6)
        turn.reset()
        self.assertAlmostEqual(turn.state()['limit_deg'], 30)
        turn.set_rate_limit(0)
        self.assertEqual(self.command(turn, 1), 0)

    def test_turn_limit_validation(self):
        turn = TurnController(.01)
        turn.set_rate_limit(180)
        self.assertAlmostEqual(turn.rate_limit, np.pi)
        self.assertEqual(ControlPanel.command(dict(name='turn_limit', value=90)), 'turn_limit 90.0')
        for value in (-1, 721, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                turn.set_rate_limit(value)
            with self.assertRaises(ValueError):
                ControlPanel.command(dict(name='turn_limit', value=value))

    def test_turn_acceleration_and_braking_are_live(self):
        turn = TurnController(.01)
        turn.set_rate_limit(90)
        turn.set_acceleration_limit(360)
        for _ in range(25):
            self.command(turn, 1)
        self.assertAlmostEqual(turn.reference, np.pi/2)
        for _ in range(25):
            self.command(turn, 0)
        self.assertAlmostEqual(turn.reference, 0)
        turn.reset()
        self.assertAlmostEqual(turn.state()['acceleration_deg'], 360)

    def test_turn_command_cap_is_live_and_preserved_on_reset(self):
        turn = TurnController(.01)
        turn.set_rate_limit(720)
        turn.set_acceleration_limit(3600)
        for _ in range(25):
            command = self.command(turn, 1)
        self.assertEqual(command, 75)
        turn.set_command_limit(100)
        self.assertEqual(self.command(turn, 1), 500)
        turn.set_command_limit(0)
        self.assertEqual(self.command(turn, 1), 0)
        turn.set_command_limit(50)
        turn.reset()
        self.assertEqual(turn.state()['command_percent'], 50)

    def test_expanded_turn_controls_validate_ranges(self):
        turn = TurnController(.01)
        for name, setter, maximum, invalid in [
            ('turn_limit', turn.set_rate_limit, 720, (-1, 721)),
            ('turn_acceleration', turn.set_acceleration_limit, 3600, (0, 3601)),
            ('turn_command_limit', turn.set_command_limit, 100, (-1, 101)),
        ]:
            setter(maximum)
            self.assertEqual(ControlPanel.command(dict(name=name, value=maximum)), f'{name} {float(maximum)}')
            for value in (*invalid, float('nan'), float('inf')):
                with self.assertRaises(ValueError):
                    setter(value)
                with self.assertRaises(ValueError):
                    ControlPanel.command(dict(name=name, value=value))

    def test_turning_does_not_cancel_forward_reference(self):
        drive = DriveController(.01, 40, 20, 2)
        obs = np.zeros((1, 162), np.float32)
        for _ in range(200):
            drive.adapt(obs, .0825, 1, turning=True)
            drive.record(obs, .0825)
        self.assertEqual(drive.reference, drive.limits.speed)

    def test_idle_does_not_change_existing_straight_drive(self):
        turn = TurnController(.01)
        self.assertEqual(self.command(turn, 0, left=10, right=20), 0)
        self.assertFalse(turn.engaged)

    def test_reset_and_balance_recovery_clear_turn(self):
        turn = TurnController(.01)
        for _ in range(100):
            self.command(turn, 1)
        self.assertEqual(self.command(turn, 1, self.drive(require_release=True)), 0)
        self.assertEqual(turn.reference, 0)
        self.assertEqual(turn.integral, 0)
        drive = DriveController(.01, 40, 20, 2)
        obs = np.zeros((1, 162), np.float32)
        obs[0, :-2].reshape(40, 4)[:, 0] = 1.4
        drive.adapt(obs, .0825, 0, turning=True)
        self.assertTrue(drive.require_release)
        obs[:] = 0
        drive.adapt(obs, .0825, 0, turning=True)
        self.assertTrue(drive.require_release)
        drive.adapt(obs, .0825, 0, turning=False)
        self.assertFalse(drive.require_release)

    def test_input_timeout_and_stale_packets_clear_both_axes(self):
        panel = ControlPanel(queue.Queue())
        try:
            with patch('controls.time.monotonic', return_value=10):
                panel.receive_drive(1, 1, -1)
                self.assertEqual(panel.inputs(), (1, -1))
                panel.receive_drive(0, 3, 0)
                panel.receive_drive(1, 2, 1)
                self.assertEqual(panel.inputs(), (0, 0))
                panel.receive_drive(0, 4, 1)
            with patch('controls.time.monotonic', return_value=10.36):
                self.assertEqual(panel.inputs(), (0, 0))
            for turn in (2, True, .5):
                with self.assertRaises(ValueError):
                    panel.command(dict(name='drive', direction=0, sequence=1, turn=turn))
        finally:
            panel.server.server_close()


class TurningPlantTests(unittest.TestCase):
    def plant(self, turning=True):
        rand = replace(Randomization(), motor_mismatch=U(0, 0, 0), fb_gain_error=U(0, 0, 0),
                       push_rate=U(0, 0, 0), change_rate=U(0, 0, 0))
        plant = Plant(1, rand, .002, np.random.default_rng(42), scale=0)
        return TurningPlant.from_plant(plant, 500) if turning else plant

    def test_pitch_and_wheel_axis_follow_heading(self):
        plant = self.plant()
        pitch = np.array([np.cos(.1), 0, np.sin(.1), 0])
        for angle in (0, np.pi/2, np.pi, -np.pi/2):
            yaw = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])
            mujoco.mju_mulQuat(plant.qpos[0, 3:7], yaw, pitch)
            np.testing.assert_allclose(plant.base_pitch(), [.2], atol=1e-12)
            np.testing.assert_allclose(plant.wheel_axis()[0], [-np.sin(angle), np.cos(angle), 0], atol=1e-12)

    def test_no_turn_matches_shared_training_plant(self):
        base, turned = self.plant(False), self.plant()
        for _ in range(30):
            base.advance(np.array([10.]), 1)
            turned.advance(np.array([10.]), 1)
        np.testing.assert_allclose(turned.d.qpos, base.d.qpos, atol=1e-8)
        np.testing.assert_allclose(turned.d.qvel, base.d.qvel, atol=1e-8)
        np.testing.assert_array_equal(turned.fb_rpm, base.fb_rpm)

    def test_fitted_motor_and_hall_match_shared_plant(self):
        from bizbot_mj import fixed
        rand = fixed(voltage_mode=1, hall_speed=1, cmd_deadzone=1,
                     cmd_deadband=23, cmd_limit=390, hall_jitter=.02,
                     motor_mismatch=0, fb_gain_error=0,
                     push_rate=0, change_rate=0, hand_rate=0)
        base = Plant(1, rand, .002, np.random.default_rng(42), scale=0)
        other = Plant(1, rand, .002, np.random.default_rng(42), scale=0)
        turned = TurningPlant.from_plant(other, 500)
        for plant in (base, turned):
            plant.reset(np.array([0]), np.zeros(1), np.zeros(1), np.zeros(1))
        for command in (10, 100, 500, -100, -500, 0):
            for _ in range(20):
                base.advance(np.array([float(command)]), 1)
                turned.advance(np.array([float(command)]), 1)
                np.testing.assert_allclose(turned.d.ctrl, base.d.ctrl, atol=1e-7)
                np.testing.assert_allclose(turned.d.qvel, base.d.qvel, atol=1e-7)
                np.testing.assert_allclose(turned.hall_est, base.hall_est, atol=1e-7)
                np.testing.assert_array_equal(turned.fb_rpm, base.fb_rpm)

    def test_differential_feedback_and_reset(self):
        plant = self.plant()
        plant.turn_command[0] = 30
        for _ in range(100):
            plant.advance(np.array([0.]), 1)
        self.assertGreater(plant.fb_rpm[1, 0], plant.fb_rpm[0, 0])
        self.assertGreater(plant.qvel[0, 5], 0)
        plant.reset(np.array([0]), np.zeros(1), np.zeros(1), np.zeros(1))
        np.testing.assert_array_equal(plant.turn_command, 0)
        np.testing.assert_array_equal(plant.turn_f, 0)
        np.testing.assert_array_equal(plant.hist_turn_cmd, 0)
        np.testing.assert_array_equal(plant.hist_turn_rpm, 0)

    def test_motor_headroom_preserves_common_command(self):
        plant = self.plant()
        plant.p['speed_ki_ratio'][:] = 0
        plant.p['spd_ki'][:] = 0
        plant.eferu_next[:] = 100
        for common in (490, -490):
            plant.u_f[:] = common
            plant.turn_f[:] = 75
            plant.advance(np.array([common]), 1)
            target = plant.d.ctrl.reshape(1, 2).T / (plant.p['k_m'] * plant.p['speed_kp'])
            commands = target / plant.p['spd_full'] * 1000
            np.testing.assert_allclose(commands.mean(axis=0), [common])
            self.assertLessEqual(np.abs(commands).max(), 500 + 1e-10)


if __name__ == '__main__':
    unittest.main()
