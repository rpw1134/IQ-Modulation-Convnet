import numpy as np
from typing import Literal

from .maps import *


class IQGenerator:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate(self, n_samples, length=256, seed=None, modulation_scheme = Literal["BPSK", "QPSK", "16QAM", "64QAM"]):
        # in case you want to use the same generator to create different datasets
        rand = self.rng
        if seed is not None:
            rand = np.random.default_rng(seed)

        # get boundaries for samples
        i_bounds, q_bounds = scheme_to_high_low_map[modulation_scheme]
        i_low, i_high = i_bounds
        q_low, q_high = q_bounds

        i_samples = rand.integers(i_low, i_high, size=(n_samples, length))
        q_samples = rand.integers(q_low, q_high, size=(n_samples, length))

        # convert into proper format for IQ
        i_samples = 2 * i_samples + 1
        if modulation_scheme != "BPSK":
            q_samples = 2 * q_samples + 1

        # stack to create distinct channels
        return np.stack((i_samples, q_samples), axis=2)

