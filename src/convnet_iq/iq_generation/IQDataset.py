from typing import Union

import torch
from numpy import ndarray
from torch.utils.data import Dataset


class IQDataset(Dataset):
    def __init__(self, data: Union[ndarray, torch.Tensor], labels: Union[ndarray, torch.Tensor], transform=None):
        """
        Args:
            data: np.ndarray of shape (n_samples, length, 2), where axis 2 holds the I sample at index 0 and the Q sample at index 1.
            labels: np.ndarray of shape (n_samples,) containing the corresponding modulation scheme labels for each IQ sequence.
        """
        self.data: torch.Tensor = torch.from_numpy(data) if isinstance(data, ndarray) else data
        self.labels: torch.Tensor = torch.from_numpy(labels) if isinstance(labels, ndarray) else labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label