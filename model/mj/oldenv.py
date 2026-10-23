"""The old numpy sim (sim/bizbot_sim) behind the same interface as the MuJoCo
env, so the same trainer can run on both. priv is the true state."""

import numpy as np

from bizbot_sim import BizBotEnv


class OldEnv(BizBotEnv):
    def __init__(self, cfg, rand, seed=None, scale=1.0):
        super().__init__(cfg, rand, seed=seed, scale=scale)
        self.priv_size = 7
        self.last_priv = self._priv()

    def _priv(self):
        t = self.plant.truth()
        return np.stack([t["pitch"] * 5.0, t["pitch_rate"], t["speed"] / 0.3, t["slope"] * 10.0,
                         t["push"] / 100.0, t["command_at_motor"] / 500.0, self.linear],
                        axis=1).astype(np.float32)

    def step(self, action):
        obs, rew, term, trunc, info = super().step(action)
        self.last_priv = self._priv()
        if info:
            info["terminal_priv"] = self.last_priv[info["done"]]
        return obs, rew, term, trunc, info
