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
            # self.layer2.add_module('{0}-{1}'.format(i, 'dropput'), nn.Dropout(0.5))
            self.layer2.add_module('{0}-{1}'.format(i, 'conv'), nn.Conv2d(F_NODE * (i - 1), F_NODE * i, Kb))
            self.layer2.add_module('{0}-{1}'.format(i, 'relu'), nn.ReLU())

        self.fc1 = nn.Sequential(
            nn.Dropout(0.2),
            # nn.Dropout(0.5),
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

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             # init.orthogonal_(m.weight)
    #             # init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 init.constant_(m.bias, 0)
    #         if isinstance(m, nn.Linear):
    #             # init.orthogonal_(m.weight)
    #             # init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 init.constant_(m.bias, 0)


class CNN_var(nn.Module):
    def __init__(self, window_size=100, N_OF_L=3):
        super(CNN_var, self).__init__()
        self.window_size = window_size
        self.N_OF_L = N_OF_L
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, padding=1, bias=True),
            # nn.BatchNorm1d(16),
            nn.LeakyReLU(),

            # nn.MaxPool1d(3, 2)
        )
        self.layer2 = nn.Sequential()
        for i in range(N_OF_L - 1):
            self.layer2.add_module('{0}-{1}'.format(i + 2, 'dropput'), nn.Dropout(0.1)),
            self.layer2.add_module('{0}-{1}'.format(i + 2, 'conv'),
                                   nn.Conv1d(in_channels=16 * (i + 1), out_channels=16 * (i + 2), kernel_size=3,
                                             padding=1,
                                             bias=True)),
            # self.layer2.add_module('{0}-{1}'.format(i + 2, 'BN'),nn.BatchNorm1d(16 * (i + 2)))
            self.layer2.add_module('{0}-{1}'.format(i + 2, 'Lrelu'), nn.LeakyReLU())
            # self.layer2.add_module('{0}-{1}'.format(i + 2, 'pooling'), nn.MaxPool1d(3, 2))

        self.fc1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(16 * self.N_OF_L * self.window_size, 32),
            nn.LeakyReLU())
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(32, N_OF_CLASSES),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # print(self.layer1.parameters())
        x = x.view(-1, 16 * self.N_OF_L * self.window_size)

        x = self.fc1(x)
        # print(x.shape)
        x = self.fc2(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.orthogonal_(m.weight)

                if m.bias is not None:
                    init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                init.orthogonal_(m.weight)
                # init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class CNN_Pooling(nn.Module):
    def __init__(self, window_size=96, N_OF_L=3):
        super(CNN_Pooling, self).__init__()
        self.window_size = window_size
        self.N_OF_L = N_OF_L
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm1d(16),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(3, 2, padding=1)
        )

        self.layer_mul = nn.Sequential()
        for i in range(N_OF_L - 1):
            self.layer_mul.add_module('{0}-{1}'.format(i + 2, 'conv1'),
                                      nn.Conv1d(in_channels=16 * (2 ** i), out_channels=16 * (2 ** i), kernel_size=3, padding=1, bias=True))
            self.layer_mul.add_module('{0}-{1}'.format(i + 2, 'BN1'), nn.BatchNorm1d(16 * (2 ** i)))
            self.layer_mul.add_module('{0}-{1}'.format(i + 2, 'Lrelu1'), nn.LeakyReLU())
            self.layer_mul.add_module('{0}-{1}'.format(i + 2, 'dropput1'), nn.Dropout(0.5))

            self.layer_mul.add_module('{0}-{1}'.format(i + 2, 'conv2'),
                                      nn.Conv1d(in_channels=16 * (2 ** i), out_channels=16 * (2 ** (i + 1)), kernel_size=3, padding=1,  bias=True))
            self.layer_mul.add_module('{0}-{1}'.format(i + 2, 'BN2'), nn.BatchNorm1d(16 * (2 ** (i + 1))))
            self.layer_mul.add_module('{0}-{1}'.format(i + 2, 'Lrelu2'), nn.LeakyReLU())

            self.layer_mul.add_module('{0}-{1}'.format(i + 2, 'dropput2'), nn.Dropout(0.5))
            self.layer_mul.add_module('{0}-{1}'.format(i + 2, 'pooling'), nn.MaxPool1d(3, 2, padding=1))

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(8 * self.window_size, 32),
            nn.LeakyReLU())
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(32, N_OF_CLASSES),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer_mul(x)
        # print(self.layer1.parameters())
        x = x.view(-1, 8 * self.window_size)

        x = self.fc1(x)
        # print(x.shape)
        x = self.fc2(x)
        return x

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv1d):
    #             init.orthogonal_(m.weight)
    #
    #             if m.bias is not None:
    #                 init.constant_(m.bias, 0)
    #         if isinstance(m, nn.Linear):
    #             init.orthogonal_(m.weight)
    #             # init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 init.constant_(m.bias, 0)


if __name__ == '__main__':
    # CNN = CNN_DMD(10).double()
    # # print(CNN)
    # input = np.random.randn(16, 1, 3, 10)
    #
    # input = torch.from_numpy(input).double()
    # # print(input.dtype)
    # output = CNN(input)
    # # print(output.shape)
    # # print(output)

    # model = CNN_var(100).double()
    # inputs = np.random.randn(16, 3, 100)
    # inputs = torch.from_numpy(inputs).double()
    # outputs = model(inputs)
    # print(outputs.shape)
    # print(outputs)


    model = CNN_Pooling(96).double()
    inputs = np.random.randn(16, 3, 96)
    inputs = torch.from_numpy(inputs).double()
    outputs = model(inputs)
    print(outputs.shape)
    # print(outputs)