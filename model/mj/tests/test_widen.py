import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppo import ACT
import numpy as np

from ppo import RunningNorm
from td3 import TanhActor, Twin, rescale_norm, widen_inputs


class WidenTests(unittest.TestCase):
    def test_widened_networks_ignore_the_new_inputs(self):
        torch.manual_seed(0)
        act = ACT['elu']
        actor, wide = TanhActor(10, (16,), act), TanhActor(11, (16,), act)
        st = actor.state_dict()
        widen_inputs(st, 10, 1, ['net.0.weight'])
        wide.load_state_dict(st)
        x = torch.randn(5, 10)
        xw = torch.cat([x, torch.randn(5, 1)], 1)
        torch.testing.assert_close(wide(xw), actor(x))

        critic, cwide = Twin(10 + 4, (16,), act), Twin(11 + 4, (16,), act)
        st = critic.state_dict()
        widen_inputs(st, 10, 1, [k for k in st if k.endswith('.0.weight')])
        cwide.load_state_dict(st)
        o, p, a = torch.randn(5, 10), torch.randn(5, 4), torch.randn(5, 1)
        ow = torch.cat([o, torch.randn(5, 1)], 1)
        for q, qw in zip(critic(torch.cat([o, p], 1), a), cwide(torch.cat([ow, p], 1), a)):
            torch.testing.assert_close(qw, q)

    def test_rescaled_norm_gives_same_input(self):
        nm = RunningNorm(3)
        nm.mean, nm.var = np.array([0.1, -0.2, 0.3]), np.array([1.0, 0.04, 2.0])
        st = rescale_norm(nm.state(), 1, 0.25)
        new = RunningNorm(3)
        new.mean, new.var = np.array(st["mean"]), np.array(st["var"])
        x = np.array([[0.5, 0.6, -0.7]])
        x_new = x.copy()
        x_new[0, 1] *= 0.25
        np.testing.assert_allclose(new(x_new), nm(x), rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
