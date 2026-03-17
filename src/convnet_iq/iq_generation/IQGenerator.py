import numpy as np
from typing import Literal, List, Tuple

from numpy import ndarray
from torch.utils.data import Dataset

from .IQDataset import IQDataset
from .maps import *


class IQGenerator:
    """Uses a seeded NumPy random generator so that datasets are reproducible.
    Supported schemes: BPSK, QPSK, 16QAM, 64QAM.
    """

    def __init__(self, seed: int = 42, scheme_distribution = (0.25,0.25,0.25,0.25)):
        """
        Args:
            seed: Seed for the default NumPy random generator. Defaults to 42.
            scheme_distribution: Percentage distribution of modulation schemes in generated datasets. Should be a list of 4 integers summing to 100, corresponding to BPSK, QPSK, 16QAM, and 64QAM respectively. Defaults to an even distribution.
        """
        self.rng = np.random.default_rng(seed)
        self.scheme_distribution = scheme_distribution
        if sum(scheme_distribution) != 1 or len(scheme_distribution) != 4:
            raise ValueError("scheme_distribution must be a list of 4 floats summing to 1.")

    def _get_scheme_boundaries(self):
        """Compute cumulative probability boundaries for each modulation scheme.

        Returns:
            Tuple of (first_bound, second_bound, third_bound) where each value
            is the upper boundary of the corresponding scheme's probability range:
            BPSK → [0, first), QPSK → [first, second), 16QAM → [second, third),
            64QAM → [third, 1].
        """
        bpsk_dist, qpsk_dist, sixteen_qam_dist, sixtyfour_qam_dist = self.scheme_distribution
        first_bound = bpsk_dist
        second_bound = first_bound + qpsk_dist
        third_bound = second_bound + sixteen_qam_dist
        return first_bound, second_bound, third_bound

    def _get_scheme_masks(self, num_samples=128):
        """Generate boolean masks assigning each sample to a modulation scheme.

        Draws a uniform random value per sample and partitions samples into
        schemes according to scheme_distribution.

        Args:
            num_samples: Number of samples to assign. Defaults to 128.

        Returns:
            Tuple of four boolean arrays (bpsk_mask, qpsk_mask, stqam_mask,
            sfqam_mask), each of shape (num_samples,).
        """
        first_bound, second_bound, third_bound = self._get_scheme_boundaries()
        uni = self.rng.random(size=(num_samples,))
        bpsk_mask = uni < first_bound
        qpsk_mask = (uni >= first_bound) & (uni < second_bound)
        stqam_mask = (uni >= second_bound) & (uni < third_bound)
        sfqam_mask = uni >= third_bound
        return bpsk_mask, qpsk_mask, stqam_mask, sfqam_mask

    def _allocate_iq_and_label_arrays(self, num_samples, length, datatype=np.int8):
        """Allocate zeroed arrays for IQ signals and symbol label indices.

        Args:
            num_samples: Number of signal sequences.
            length: Number of symbols per sequence.

        Returns:
            Tuple of (iq_arr, symbol_indices_arr) where iq_arr has shape
            (num_samples, length, 2) with dtype int8, and symbol_indices_arr
            has shape (num_samples, length) with dtype uint8.
        """
        iq_arr = np.zeros(shape=(num_samples, length, 2), dtype=datatype)
        symbol_indices_arr = np.zeros(shape=(num_samples, length), dtype=np.uint8)
        return iq_arr, symbol_indices_arr

    def _generate_mask_scheme_pairs(self, num_samples):
        """Pair each scheme's boolean mask with its scheme name.

        Args:
            num_samples: Number of samples to assign across schemes.

        Returns:
            List of (mask, scheme) tuples where mask is a boolean array of
            shape (num_samples,) and scheme is the corresponding scheme string.
        """
        bpsk_mask, qpsk_mask, stqam_mask, sfqam_mask = self._get_scheme_masks(num_samples=num_samples)
        pairs: List [Tuple[ndarray, Literal["BPSK", "QPSK", "16QAM", "64QAM"]]] = [
            (bpsk_mask, "BPSK"),
            (qpsk_mask, "QPSK"),
            (stqam_mask, "16QAM"),
            (sfqam_mask, "64QAM"),
        ]
        return pairs

    def generate_signals(self, n_samples=128, length=256, seed=None, modulation_scheme: Literal["BPSK", "QPSK", "16QAM", "64QAM"] = "BPSK"):
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

    def generate_softmax_indices_for_signals(self, iq_signals, modulation_scheme: Literal["BPSK", "QPSK", "16QAM", "64QAM"] = "BPSK"):
        """Map IQ signal amplitudes to global softmax class indices.

        Converts raw I and Q amplitude values to table indices using per-scheme
        normalization, then looks up the global class index for each (I, Q) pair.

        Args:
            iq_signals: np.ndarray of shape (n_samples, length, 2) as returned
                by generate_signals.
            modulation_scheme: One of "BPSK", "QPSK", "16QAM", or "64QAM".

        Returns:
            np.ndarray of shape (n_samples, length) with dtype uint8 containing
            global class indices in the range defined by SCHEME_OFFSETS.
        """
        index_table = scheme_to_index_table_map[modulation_scheme]
        i_offset, i_step, q_offset, q_step = SCHEME_NORMALIZATION[modulation_scheme]
        i_idx = (iq_signals[:, :, 0] + i_offset) // i_step
        q_idx = (iq_signals[:, :, 1] + q_offset) // q_step
        return index_table[i_idx, q_idx]

    def generate_dataset(self, num_samples=128, length=256, datatype=np.int8):
        """Generate a mixed-scheme IQDataset ready for training.

        Samples are assigned to modulation schemes according to
        scheme_distribution, then IQ signals and their corresponding global
        softmax label indices are generated for each scheme.

        Args:
            num_samples: Total number of signal sequences in the dataset.
                Defaults to 128.
            length: Number of symbols per sequence. Defaults to 256.

        Returns:
            IQDataset with data of shape (num_samples, length, 2) and labels
            of shape (num_samples, length) containing global class indices.
        """
        # allocate memory for the dataset
        iq_arr, symbol_indices_arr = self._allocate_iq_and_label_arrays(num_samples=num_samples, length=length, datatype=datatype)

        # map masks to schemes for generation purposes
        mask_scheme_pairs = self._generate_mask_scheme_pairs(num_samples=num_samples)

        # loop over each mask and set to a generate signal. Set labels based on the scheme
        for mask, scheme in mask_scheme_pairs:
            # number of samples to generate for this scheme is the number of True values in the mask
            count = mask.sum()
            if count == 0:
                continue
            # set those rows to IQ signals
            iq_arr[mask] = self.generate_signals(n_samples=count, length=length, modulation_scheme=scheme)
            symbol_indices_arr[mask] = self.generate_softmax_indices_for_signals(iq_arr[mask], modulation_scheme=scheme)

        return IQDataset(data=iq_arr, labels=symbol_indices_arr)



