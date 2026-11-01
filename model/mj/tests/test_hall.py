import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'sim')]
from bizbot_mj import fixed
from bizbot_mj.plant import Plant, RPM_PER_RAD_S


class HallTests(unittest.TestCase):
    def plant(self, n=1):
        return Plant(n, fixed(hall_speed=1, hall_edges=90, hall_jitter=0,
                              hall_timeout=.1, fb_lag=.002,
                              motor_mismatch=0, push_rate=0,
                              change_rate=0, hand_rate=0),
                     .002, np.random.default_rng(17), scale=0)

    def test_opposite_wheel_motion_is_not_lost_in_mean(self):
        pl = self.plant()
        pl.qvel[:, 7:9] = [1., -1.]
        for _ in range(200):
            pl._update_wheel_hall()
            pl.t += pl.dt
        np.testing.assert_allclose(pl.wheel_hall_est, [[1., -1.]], atol=1e-10)
        np.testing.assert_allclose(pl.hist_wheel_rpm[pl.ptr], [[RPM_PER_RAD_S, -RPM_PER_RAD_S]], atol=1e-10)

    def test_timeout_is_independent_per_wheel(self):
        pl = self.plant()
        pl.qvel[:, 7:9] = 2.
        for _ in range(100):
            pl._update_wheel_hall()
            pl.t += pl.dt
        pl.qvel[:, 7] = 0
        for _ in range(100):
            pl._update_wheel_hall()
            pl.t += pl.dt
        self.assertEqual(pl.wheel_hall_est[0, 0], 0)
        self.assertAlmostEqual(pl.wheel_hall_est[0, 1], 2.)

    def test_partial_reset_clears_only_selected_histories(self):
        pl = self.plant(2)
        pl.hist_wheel_rpm[:] = 23
        pl.wheel_hall_est[:] = 12
        pl.reset(np.array([0]), np.zeros(1), np.zeros(1), np.zeros(1))
        np.testing.assert_array_equal(pl.hist_wheel_rpm[:, 0], 0)
        np.testing.assert_array_equal(pl.hist_wheel_rpm[:, 1], 23)
        np.testing.assert_array_equal(pl.wheel_hall_est[1], 12)
        self.assertFalse(np.allclose(pl.wheel_hall_pos[0, 0], pl.wheel_hall_pos[0, 1]))

    def test_feedback_rounds_each_wheel_before_averaging(self):
        pl = self.plant()
        pl.hist_wheel_rpm[:] = [10.2, 11.2]
        pl.p['fb_delay'][:] = 1
        pl.p['fb_gain'][:] = 1
        pl.p['fb_min_rpm'][:] = 0
        pl.fb_next[:] = 0
        pl.advance(np.zeros(1), 1)
        np.testing.assert_array_equal(pl.fb_rpm[:, 0], [10, 11])
        self.assertEqual(pl.fb_rpm[:, 0].mean(), 10.5)


if __name__ == '__main__':
    unittest.main()
