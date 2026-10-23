"""stable-baselines3 VecEnv around the sim's numpy BizBotEnv.

BizBotEnv already steps all robots at once and resets fallen ones itself, so
this only translates its outputs into what SB3 expects.
"""

import sys
from collections import deque
from pathlib import Path

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim"))
from bizbot_sim import BizBotEnv  # noqa: E402


class BizBotVecEnv(VecEnv):
    def __init__(self, cfg, rand, seed=None, scale=0.0):
        self.env = BizBotEnv(cfg, rand, seed=seed, scale=scale)
        obs_space = spaces.Box(-np.inf, np.inf, (self.env.obs_size,), np.float32)
        act_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
        self.render_mode = None
        super().__init__(cfg.n_envs, obs_space, act_space)
        self.recent_fell = deque(maxlen=2000)   # last finished episodes: fell or ran out of time
        self._actions = None

    @property
    def scale(self):
        return self.env.scale

    @scale.setter
    def scale(self, value):
        self.env.scale = value

    def reset(self):
        return self.env.reset()

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        obs, reward, terminated, truncated, info = self.env.step(self._actions)
        dones = terminated | truncated
        infos = [{} for _ in range(self.num_envs)]
        if info:
            for k, i in enumerate(info["done"]):
                infos[i] = {
                    "terminal_observation": info["terminal_obs"][k],
                    "TimeLimit.truncated": bool(truncated[i] and not terminated[i]),
                    "episode": {"r": float(info["episode_return"][k]),
                                "l": int(info["episode_length"][k])},
                }
            self.recent_fell.extend(bool(f) for f in info["fell"])
        return obs, reward, dones, infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name)] * len(self._get_indices(indices))

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [getattr(self, method_name)(*args, **kwargs)] * len(self._get_indices(indices))

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._get_indices(indices))
