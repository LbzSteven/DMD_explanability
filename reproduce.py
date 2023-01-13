import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from window import window_oper
from CNN_torch import CNN_DMD
from dataset import DMDDataset
from result_record import csv_writer
import torchvision.transforms as transforms
import torch.optim as optim

DMD_group_number = ['990012', '990015', '990016', '990023008', '990023010', '990023015']
TD_group_number = ['990014', '990017', '990018', '990023003', '990023011', '990023014']
all_group_number = ['990012', '990014', '990015', '990016', '990017', '990018', '990023008', '990023010', '990023015',
                    '990023003', '990023011', '990023014']
low_sample_rate = ['990012', '990014', '990015', '990016', '990017', '990018']
high_sample_rate = ['990023003', '990023008', '990023010', '990023011', '990023014', '990023015']

# WINDOW_SIZE = 33
# WINDOW_STEP = 33
LEARN_RATE = 0.001
BATCH_SIZE = 128
# EPOCH = 100000
NUM_WORKERS = 0

# 12 fold training and test
# transforms = transforms.Compose(
#     [
#         transforms.ToTensor(),
#     ]
# )
transforms = None


def CNN_debug():
    EPOCH = 200
    WINDOW_STEP = 50 #33
    WINDOW_SIZE = 50 #33
    correct = 0
    total = len(all_group_number)
    paitent_makers, window_labels, window_data = window_oper(all_group_number, WINDOW_SIZE, WINDOW_STEP)
    window_labels = np.array(window_labels)
    window_data = np.array(window_data)
    print('current: EPOCH %d W_SIZE %d W_STEP %d' % (EPOCH, WINDOW_SIZE, WINDOW_STEP))

    for number in all_group_number:
        test_acc_epoch = []
        train_acc_epoch = []
        loss_epoch = []
        testing_idx = [i for i, x in enumerate(paitent_makers) if x == number]
        training_idx = [i for i, x in enumerate(paitent_makers) if x != number]

        testing_labels = window_labels[testing_idx]
        testing_data = window_data[testing_idx, :]
        training_labels = window_labels[training_idx]
        training_data = window_data[training_idx, :]

        trainset = DMDDataset(training_labels, training_data, transforms)
        testset = DMDDataset(testing_labels, testing_data, transforms)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                                  shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                                 shuffle=False)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        net = CNN_DMD(WINDOW_SIZE).float().to(device)
        optimizer = optim.Adam(net.parameters())
        loss_function = nn.CrossEntropyLoss()
        # print('patient: ', number, ' train start')
        for epoch in range(EPOCH):
            running_loss = 0
            for i, (sample) in enumerate(trainloader, 0):
                # get the inputs

                labels, data = sample['label'].to(device).to(torch.int64), sample['data'].to(device).float()

                optimizer.zero_grad()
                outputs = net(data)

                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            # print(running_loss)
            window_correct_test = 0
            window_total_test = 0
            window_correct_train = 0
            window_total_train = 0
            # print('patient: ', number, ' train finished')
            with torch.no_grad():
                for i, sample in enumerate(testloader, 0):
                    labels, data = sample['label'].to(device).to(torch.int64), sample['data'].to(device).float()

                    outputs = net(data)
                    _, predicted = torch.max(outputs.data, 1)
                    window_total_test += labels.size(0)
                    # print(predicted)
                    # print(labels)
                    # print(window_total_test)
                    window_correct_test += (predicted == labels).sum().item()

                for i, sample in enumerate(trainloader, 0):
                    labels, data = sample['label'].to(device).to(torch.int64), sample['data'].to(device).float()

                    outputs = net(data)
                    _, predicted = torch.max(outputs.data, 1)
                    window_total_train += labels.size(0)

                    window_correct_train += (predicted == labels).sum().item()
            correct_percentage_test = window_correct_test / window_total_test
            correct_percentage_train = window_correct_train / window_total_train

            print('%s: epoch %d train per:%.3f,test per:%.3f,run loss %.3f' % (
            number, epoch, correct_percentage_train, correct_percentage_test, running_loss))
            train_acc_epoch.append(correct_percentage_train)
            test_acc_epoch.append(correct_percentage_test)
            loss_epoch.append(running_loss)
        plt.figure(figsize=(18, 15))
        plt.ylabel('Magnitude on ', fontsize=20)
        ax = plt.subplot(311)
        ax.plot(range(EPOCH), train_acc_epoch, label='train_acc_epoch')
        ax.set_xlabel('epoch')
        ax.set_ylabel('train_acc_epoch')
        ax.grid()
        ax.legend()

        ax = plt.subplot(312)
        ax.plot(range(EPOCH), test_acc_epoch, label='train_acc_epoch')
        ax.set_xlabel('epoch')
        ax.set_ylabel('train_acc_epoch')
        ax.grid()
        ax.legend()

        ax = plt.subplot(313)
        ax.plot(range(EPOCH), loss_epoch, label='loss')
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss_epoch')
        ax.grid()
        ax.legend()

        plt.savefig(os.path.join('visualize/CNN_acc_loss', number))
CNN_debug()
exit()

j = 0


def CNN_debug():
    EPOCH = 10
    WINDOW_SIZE = 33
    WINDOW_STEP = 33
    correct = 0
    total = len(all_group_number)
    paitent_makers, window_labels, window_data = window_oper(all_group_number, WINDOW_SIZE, WINDOW_STEP)
    window_labels = np.array(window_labels)
    window_data = np.array(window_data)
    # print('current: EPOCH %d W_SIZE %d W_STEP %d' % (EPOCH, WINDOW_SIZE, WINDOW_STEP))
    for number in all_group_number:
        testing_idx = [i for i, x in enumerate(paitent_makers) if x == number]
        training_idx = [i for i, x in enumerate(paitent_makers) if x != number]

        testing_labels = window_labels[testing_idx]
        testing_data = window_data[testing_idx, :]
        training_labels = window_labels[training_idx]
        training_data = window_data[training_idx, :]

        trainset = DMDDataset(training_labels, training_data, transforms)
        testset = DMDDataset(testing_labels, testing_data, transforms)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                                  shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                                 shuffle=False)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        net = CNN_DMD(WINDOW_SIZE).float().to(device)
        optimizer = optim.Adam(net.parameters())
        loss_function = nn.CrossEntropyLoss()
        # print('patient: ', number, ' train start')
        for epoch in range(EPOCH):
            running_loss = 0
            for i, (sample) in enumerate(trainloader, 0):
                # get the inputs

                labels, data = sample['label'].to(device).to(torch.int64), sample['data'].to(device).float()

                optimizer.zero_grad()
                outputs = net(data)

                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            # print(running_loss)
            window_correct = 0
            window_total = 0
            train_total = 0
            train_correct = 0
            # print('patient: ', number, ' train finished')
            with torch.no_grad():
                for i, sample in enumerate(testloader, 0):
                    labels, data = sample['label'].to(device).to(torch.int64), sample['data'].to(device).float()

                    outputs = net(data)
                    _, predicted = torch.max(outputs.data, 1)
                    window_total += labels.size(0)

                    window_correct += (predicted == labels).sum().item()
                correct_percentage_test = window_correct / window_total

                for i, sample in enumerate(trainloader, 0):
                    labels, data = sample['label'].to(device).to(torch.int64), sample['data'].to(device).float()

                    outputs = net(data)
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += labels.size(0)

                    train_correct += (predicted == labels).sum().item()

                correct_percentage = train_correct / train_total
                print('%s: train correct percentage:%.3f test correct percentage:%.3f' % (
                number, correct_percentage, correct_percentage_test))
        if correct_percentage_test > 0.5:
            correct += 1

    csv_writer(EPOCH, WINDOW_STEP, WINDOW_SIZE, 'bs_' + str(BATCH_SIZE) + '_lr_' + str(LEARN_RATE),
               (correct / total))


CNN_debug()
exit()

for EPOCH in [100, 500, 1000, 5000, 10000, 50000]:
    for WINDOW_SIZE in [30, 33, 90, 100]:
        for WINDOW_STEP in [WINDOW_SIZE, int(WINDOW_SIZE / 2), int(WINDOW_SIZE / 3)]:
            correct = 0
            total = len(all_group_number)
            paitent_makers, window_labels, window_data = window_oper(all_group_number, WINDOW_SIZE, WINDOW_STEP)
            window_labels = np.array(window_labels)
            window_data = np.array(window_data)
            print('current: EPOCH %d W_SIZE %d W_STEP %d' % (EPOCH, WINDOW_SIZE, WINDOW_STEP))
            for number in all_group_number:
                j += 1
                testing_idx = [i for i, x in enumerate(paitent_makers) if x == number]
                training_idx = [i for i, x in enumerate(paitent_makers) if x != number]

                testing_labels = window_labels[testing_idx]
                testing_data = window_data[testing_idx, :]
                training_labels = window_labels[training_idx]
                training_data = window_data[training_idx, :]

                trainset = DMDDataset(training_labels, training_data, transforms)
                testset = DMDDataset(testing_labels, testing_data, transforms)

                trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                                          shuffle=True)
                testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                                         shuffle=False)

                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

                net = CNN_DMD(WINDOW_SIZE).float().to(device)
                optimizer = optim.Adam(net.parameters())
                loss_function = nn.CrossEntropyLoss()
                # print('patient: ', number, ' train start')
                for epoch in range(EPOCH):
                    running_loss = 0
                    for i, (sample) in enumerate(trainloader, 0):
                        # get the inputs

                        labels, data = sample['label'].to(device).to(torch.int64), sample['data'].to(device).float()

                        optimizer.zero_grad()
                        outputs = net(data)

                        loss = loss_function(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                    # print(running_loss)
                window_correct = 0
                window_total = 0
                # print('patient: ', number, ' train finished')
                with torch.no_grad():
                    for i, sample in enumerate(testloader, 0):
                        labels, data = sample['label'].to(device).to(torch.int64), sample['data'].to(device).float()

                        outputs = net(data)
                        _, predicted = torch.max(outputs.data, 1)
                        window_total += labels.size(0)

                        window_correct += (predicted == labels).sum().item()
                        # print(predicted)
                        # print(labels)
                        # print(window_correct)
                        # print(window_total)
                correct_percentage = window_correct / window_total

                if correct_percentage > 0.5:
                    correct += 1
                    # print(' patient: %s predict correct, percentage:%.2f,loss:%.5f' % (number,  correct_percentage, running_loss))
                # else:
                # print(' patient: %s predict wrong, percentage:%.2f,loss:%.5f' % (number,  correct_percentage, running_loss))

                # print('total correct percentage:', correct / total)
            csv_writer(EPOCH, WINDOW_STEP, WINDOW_SIZE, 'bs_' + str(BATCH_SIZE) + '_lr_' + str(LEARN_RATE),
                       (correct / total))
