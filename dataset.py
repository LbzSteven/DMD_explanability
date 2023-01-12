import torch
import numpy as np
from torch.utils.data import Dataset
import math


class DMDDataset(Dataset):

    def __init__(self, labels, data, transform=None):

        self.labels = labels
        self.data = data[:, np.newaxis, :, :]
        self.transform = transform

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'label': self.labels[idx], 'data': self.data[idx, :, :]}

        if self.transform:
            sample = self.transform(sample)

        return sample
