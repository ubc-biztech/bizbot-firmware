import sys
import unittest
from pathlib import Path
import mujoco
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'sim'))
from bizbot_mj import fixed
from bizbot_mj.plant import Plant


class HeadingTests(unittest.TestCase):
    def plant(self):
        return Plant(1,fixed(push_rate=0,change_rate=0,hand_rate=0,
                             motor_mismatch=0,fb_gain_error=0,imu_noise_deg=0,
                             accel_noise=0,hall_speed=1),.001,np.random.default_rng(9),0)

    def test_pitch_is_independent_of_yaw(self):
        p=self.plant()
        pitch=np.array([np.cos(.1),0,np.sin(.1),0])
        for angle in (0,np.pi/2,np.pi,-np.pi/2):
            yaw=np.array([np.cos(angle/2),0,0,np.sin(angle/2)])
            mujoco.mju_mulQuat(p.qpos[0,3:7],yaw,pitch)
            np.testing.assert_allclose(p.base_pitch(),[.2],atol=1e-12)
            np.testing.assert_allclose(p.wheel_axis()[0],[-np.sin(angle),np.cos(angle),0],atol=1e-12)

    def test_rotated_robot_has_same_forward_dynamics(self):
        a,b=self.plant(),self.plant()
        for p in (a,b):
            p.reset(np.array([0]),np.array([.03]),np.zeros(1),np.zeros(1))
        q=b.qpos[0,3:7].copy()
        mujoco.mju_mulQuat(b.qpos[0,3:7],np.array([np.sqrt(.5),0,0,np.sqrt(.5)]),q)
        for _ in range(100):
            a.advance(np.array([30.]),1);b.advance(np.array([30.]),1)
        np.testing.assert_allclose(a.base_pitch(),b.base_pitch(),atol=1e-6)
        np.testing.assert_allclose(a.speed(),b.speed(),atol=1e-6)
        np.testing.assert_allclose(a.fwd_acc,b.fwd_acc,atol=1e-5)
        np.testing.assert_array_equal(a.fb_rpm,b.fb_rpm)


if __name__=='__main__':unittest.main()
