import torch
import torch.nn as nn
import numpy as np

N_OF_CLASSES = 2


class CNN_DMD(nn.Module):
    def __init__(self, window_size, N_OF_L=1, F_NODE=16, Ka=2, Kb=2):
        super(CNN_DMD, self).__init__()
        self.N_OF_L = N_OF_L
        self.F_NODE = F_NODE
        self.window_size = window_size
        self.layer1 = nn.Sequential(nn.Conv2d(1, F_NODE, Ka, bias=True),
                                    nn.ReLU())
        self.layer2 = nn.Sequential()
        for i in range(2, N_OF_L + 2):
            self.layer2.add_module('{0}-{1}'.format(i, 'dropput'), nn.Dropout(0.1))
            self.layer2.add_module('{0}-{1}'.format(i, 'conv'), nn.Conv2d(F_NODE * (i - 1), F_NODE * i, Kb))
            self.layer2.add_module('{0}-{1}'.format(i, 'relu'), nn.ReLU())

        self.fc1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear((N_OF_L + 1) * F_NODE * (window_size - N_OF_L - 1), 2 + 2 * F_NODE),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2 + 2 * F_NODE, N_OF_CLASSES),
            nn.ReLU())

    def forward(self, x):
        x = self.layer2(self.layer1(x))
        x = x.view(-1, (1 + 1) * self.F_NODE * (self.window_size - 1 - 1))
        x = self.fc2(self.fc1(x))
        return x


if __name__ == '__main__':
    CNN = CNN_DMD(10).double()
    print(CNN)
    input = np.random.randn(16, 1, 3, 10)

    input = torch.from_numpy(input).double()
    print(input.dtype)
    output = CNN(input)
    print(output.shape)
