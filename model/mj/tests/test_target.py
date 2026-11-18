import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'sim'))
from bizbot_mj.env import BalanceEnv, EnvConfig
from bizbot_mj.params import Randomization
from bizbot_mj.plant import RPM_PER_RAD_S


def make(**kw):
    return BalanceEnv(EnvConfig(n_envs=8, history=4, **kw), Randomization(), seed=1, scale=0.0)


class TargetTests(unittest.TestCase):
    def test_off_keeps_observation(self):
        env = make()
        self.assertEqual(env.obs_size, 4 * 4 + 2)
        self.assertEqual(env.reset().shape, (8, 18))

    def test_on_appends_target(self):
        env = make(target_rpm_max=100.0)
        self.assertEqual(env.obs_size, 4 * 4 + 3)
        env.reset()
        env.target[:] = 60.0
        obs, *_ = env.step(np.zeros(8))
        np.testing.assert_allclose(obs[:, -1], 1.0)

    def test_targets_change_within_limits(self):
        env = make(target_rpm_max=100.0, target_hold_min_s=0.05, target_hold_max_s=0.1, max_tilt_deg=1e9)
        env.reset()
        seen = []
        for _ in range(300):
            env.step(np.zeros(8))
            seen.append(env.target.copy())
        seen = np.array(seen)
        self.assertTrue(np.all(np.abs(seen) <= 100.0))
        self.assertTrue(np.any(seen == 0.0) and np.any(seen != 0.0))

    def test_integral_counts_rotation_relative_to_target(self):
        env = make(target_rpm_max=100.0, target_hold_min_s=1e9, target_hold_max_s=1e9, max_tilt_deg=1e9)
        env.reset()
        env.target[:] = 50.0
        before = env.integral.copy()
        env.step(np.full(8, 0.2))
        rpm = env._wheel_rpm()
        expected = before * np.exp(-0.01 / 20.0) - (rpm - 50.0) / 60.0 * 0.01
        np.testing.assert_allclose(env.integral, np.clip(expected, -2, 2), atol=1e-9)

    def test_wheel_rpm_and_ground_speed_agree_in_sign(self):
        env = make(max_tilt_deg=1e9)
        env.reset()
        for _ in range(30):
            env.step(np.full(8, 0.3))
        rpm, v = env._wheel_rpm(), env.plant.truth()['speed']
        moving = np.abs(rpm) > 20
        self.assertTrue(moving.any())
        np.testing.assert_array_equal(np.sign(rpm[moving]), np.sign(v[moving]))
        # target_speed uses the same conversion: rpm -> m/s at the wheel radius
        r = env.plant.p['wheel_radius']
        np.testing.assert_allclose(rpm / RPM_PER_RAD_S * r, v, rtol=0.5, atol=0.05)



class RampTests(unittest.TestCase):
    def test_target_ramps_to_goal(self):
        env = make(target_rpm_max=100.0, target_accel_rpm_s=200.0, target_hold_min_s=1e9,
                   target_hold_max_s=1e9, max_tilt_deg=1e9)
        env.reset()
        env.target_goal[:] = 100.0
        for _ in range(10):
            env.step(np.zeros(8))
        np.testing.assert_allclose(env.target, 20.0)
        for _ in range(100):
            env.step(np.zeros(8))
        np.testing.assert_allclose(env.target, 100.0)


if __name__ == '__main__':
    unittest.main()
