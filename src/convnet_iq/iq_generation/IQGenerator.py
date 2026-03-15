import numpy as np
from .maps import *


class IQGenerator:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_bspk(self, n_samples, length=256, seed=None):
        rand = self.rng
        if seed is not None:
            rand = np.random.default_rng(seed)
        i_samples = rand.integers(low=0, high=2, size=(n_samples, length))
        i_vals = 2 * i_samples - 1
        q_vals = np.zeros_like(i_vals)

        # stack to create shape (n_samples, length, 2) where [:,:,0] = I and [:,:,1] = Q
        return np.stack((i_vals, q_vals), axis=2)
