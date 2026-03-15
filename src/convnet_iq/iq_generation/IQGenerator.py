import numpy as np
from typing import Literal

from .maps import *


class IQGenerator:
    """Uses a seeded NumPy random generator so that datasets are reproducible.
    Supported schemes: BPSK, QPSK, 16QAM, 64QAM.
    """

    def __init__(self, seed: int = 42, scheme_distribution = [25,25,25,25]):
        """
        Args:
            seed: Seed for the default NumPy random generator. Defaults to 42.
            scheme_distribution: Percentage distribution of modulation schemes in generated datasets. Should be a list of 4 integers summing to 100, corresponding to BPSK, QPSK, 16QAM, and 64QAM respectively. Defaults to an even distribution.
        """
        self.rng = np.random.default_rng(seed)
        self.scheme_distribution = scheme_distribution
        if sum(scheme_distribution) != 100 or len(scheme_distribution) != 4:
            raise ValueError("scheme_distribution must be a list of 4 integers summing to 100.")

    def generate_signals(self, n_samples, length=256, seed=None, modulation_scheme = Literal["BPSK", "QPSK", "16QAM", "64QAM"]):
        """Randomly draws constellation point indices, maps them to odd-integer
        amplitude levels (e.g. ±1 for BPSK, ±1/±3 for QPSK), and stacks I
        and Q channels.

        Args:
            n_samples: Number of independent IQ sequences to generate.
            length: Number of symbols per sequence. Defaults to 256.
            seed: Optional per-call seed. When provided, a fresh generator is
                created for this call only, leaving the instance generator
                unchanged. Defaults to None (use the instance generator).
            modulation_scheme: One of "BPSK", "QPSK", "16QAM", or "64QAM".

        Returns:
            np.ndarray of shape (n_samples, length, 2), where axis 2 holds
            the I sample at index 0 and the Q sample at index 1.
        """
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

    def generate_softmax_indices_for_signals(self, iq_signals, modulation_scheme = Literal["BPSK", "QPSK", "16QAM", "64QAM"]):
        index_table = scheme_to_index_table_map[modulation_scheme]
        i_offset, i_step, q_offset, q_step = SCHEME_NORMALIZATION[modulation_scheme]
        i_idx = (iq_signals[:, :, 0] + i_offset) // i_step
        q_idx = (iq_signals[:, :, 1] + q_offset) // q_step
        return index_table[i_idx, q_idx]

    def generate_dataset(self):
        pass

    @property
    def data_loader(self):
        pass

