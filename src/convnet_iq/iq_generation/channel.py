import torch

from .IQDataset import IQDataset
import numpy as np
from scipy.signal import butter, sosfilt


def add_noise(signal: IQDataset,
              snr_db: float = None,
              rng: np.random.Generator = None) -> IQDataset:
    """Add AWGN (Additive White Gaussian Noise) to an IQ dataset.

    Noise power is derived per signal from its measured power and the target
    SNR, then split equally across I and Q channels.

    Args:
        signal: IQDataset whose data tensor has shape (B, L, 2).
        snr_db: Signal-to-noise ratio in decibels. If None, sampled uniformly
            from [5, 15] dB.
        rng: NumPy random generator. If None, a fresh default generator is used.

    Returns:
        New IQDataset with float32 noisy signals and the original labels.
    """
    if rng is None:
        rng = np.random.default_rng()
    if snr_db is None:
        snr_db = np.random.uniform(5, 15)

    # find signal power
    signal_data = signal.data.numpy()
    signal_power = np.mean(np.sum(signal_data**2, axis=2, keepdims=True), axis=1, keepdims=True)

    # find noise power and std deviation
    noise_powers = signal_power / (10 ** (snr_db / 10))
    noise_std = np.sqrt(noise_powers / 2)
    noise = rng.normal(0, noise_std, signal_data.shape)

    return IQDataset(signal_data + noise, signal.labels)


def add_interference(signal: IQDataset,
                     interference_signal: IQDataset,
                     interference_ratio: float = None) -> IQDataset:
    """Add a scaled interfering signal to an IQ dataset.

    Models a second transmitter whose signal is received simultaneously.
    The interference is added directly to the IQ samples; labels are unchanged
    since they reflect the intended transmitted symbols.

    Args:
        signal: IQDataset to corrupt, with data shape (B, L, 2).
        interference_signal: IQDataset used as the interferer. Must have the
            same shape as signal.
        interference_ratio: Scaling factor applied to the interferer before
            addition. If None, sampled uniformly from [0, 0.3].

    Returns:
        New IQDataset with the combined signal and the original labels.
    """
    if interference_ratio is None:
        interference_ratio = np.random.uniform(0, 0.3)

    scaled_interference = interference_signal.data * interference_ratio
    combined_signal = signal.data + scaled_interference
    return IQDataset(combined_signal, signal.labels)


def apply_low_pass_filter(signal: IQDataset) -> IQDataset:
    """Apply a 4th-order Butterworth low-pass filter to an IQ dataset.

    Models the band-limiting effect of real hardware (antennas, amplifiers,
    cables). The filter is applied along the time/symbol axis (axis=1),
    causing adjacent symbols to bleed into each other (inter-symbol
    interference). Uses second-order sections (SOS) for numerical stability.

    Args:
        signal: IQDataset whose data tensor has shape (B, L, 2).

    Returns:
        New IQDataset with float32 filtered signals and the original labels.
    """
    # TODO: COME BACK AND RESEARCH

    # Convert to numpy
    np_data = signal.data.numpy()

    # gen filter
    sos = butter(N=4, Wn=0.4, btype='low', analog=False, output='sos')
    filtered = sosfilt(sos, np_data, axis=1)

    return IQDataset(torch.from_numpy(filtered).to(dtype=torch.float32), signal.labels)
