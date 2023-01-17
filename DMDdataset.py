import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
import math


class DMDDataset(Dataset):

    def __init__(self, labels, data, transform=None, dimension=2):

        self.labels = labels
        if dimension == 2:
            self.data = data[:, np.newaxis, :, :]
        else:
            self.data = data[:, :, :]
        self.transform = transform

        # self.max, self.min = self.get_min_max(data)
        # X: vertical
        # Y: mediolateral
        # Z: anteroposterior

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'label': self.labels[idx], 'data': self.data[idx, :, :]}

        # if self.transform:
        #     sample = self.transform(sample)

        return sample
