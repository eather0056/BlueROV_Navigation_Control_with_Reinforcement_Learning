"""Bounded replay storing every observation modality, including next SBES."""
from collections import deque
import numpy as np
from .environment import KEYS


class Replay:
    def __init__(self, capacity, rng):
        self.data = deque(maxlen=capacity)
        self.rng = rng

    def __len__(self):
        return len(self.data)

    def add(self, obs, action, reward, nxt, terminated):
        # Own copies: environments may reuse their arrays.
        clone = lambda x: {k: np.array(x[k], copy=True) for k in KEYS}
        self.data.append((clone(obs), np.array(action, copy=True), float(reward),
                          clone(nxt), bool(terminated)))

    def sample(self, size):
        if size > len(self):
            raise ValueError("Not enough replay transitions")
        rows = [self.data[int(i)] for i in self.rng.choice(len(self), size=size, replace=False)]
        obs = {k: np.stack([r[0][k] for r in rows]) for k in KEYS}
        nxt = {k: np.stack([r[3][k] for r in rows]) for k in KEYS}
        return obs, np.stack([r[1] for r in rows]), np.array([[r[2]] for r in rows]), nxt, np.array([[r[4]] for r in rows], dtype=np.float32)
