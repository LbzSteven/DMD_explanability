import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
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
            # nn.Linear((N_OF_L + 1) * F_NODE * (window_size - N_OF_L - 1), 100),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2 + 2 * F_NODE, N_OF_CLASSES),
            # nn.Linear(100, N_OF_CLASSES),
            nn.Softmax(dim=1)
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer2(self.layer1(x))
        # print(self.layer1.parameters())
        x = x.view(-1, (1 + 1) * self.F_NODE * (self.window_size - 1 - 1))

        # x = self.fc2(self.fc1(x))
        x = self.fc1(x)
        # print(x.shape)
        x = self.fc2(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # init.orthogonal_(m.weight)
                # init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            if isinstance(m,nn.Linear):
                # init.orthogonal_(m.weight)
                # init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
if __name__ == '__main__':
    CNN = CNN_DMD(10).double()
    # print(CNN)
    input = np.random.randn(16, 1, 3, 10)

    input = torch.from_numpy(input).double()
    # print(input.dtype)
    output = CNN(input)
    # print(output.shape)
    # print(output)
