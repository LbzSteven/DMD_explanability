import os
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from window import window_oper, window_oper_HS_3windows
from CNN_torch import CNN_DMD, CNN_var, CNN_Pooling, one_axis_CNN
from DMDdataset import DMDDataset
from result_record import csv_writer
# import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
from constants import DMD_group_number, TD_group_number, all_group_number, low_sample_rate, high_sample_rate, \
    TD_group_number_30, DMD_group_number_30, all_group_number_30

# WINDOW_SIZE = 33
# WINDOW_STEP = 33
LEARN_RATE = 0.0001
BATCH_SIZE = 128
# EPOCH = 100000
NUM_WORKERS = 8


# np.random.seed(1)

def normalize(window_data_train, window_data_test):
    t_train = np.moveaxis(window_data_train, 1, 2)
    t_test = np.moveaxis(window_data_test, 1, 2)
    max_3_axis = np.max(t_train.reshape(-1, 3), axis=0)
    min_3_axis = np.min(t_train.reshape(-1, 3), axis=0)
    for i in range(3):
        t_train[:, :, i] = (t_train[:, :, i] - min_3_axis[i]) / (max_3_axis[i] - min_3_axis[i])
        t_test[:, :, i] = (t_test[:, :, i] - min_3_axis[i]) / (max_3_axis[i] - min_3_axis[i])
        # t_train[:, :, i] = (t_train[:, :, i] + 3) / (3 + 3)
    window_data_train = np.moveaxis(t_train, 1, 2)
    window_data_test = np.moveaxis(t_test, 1, 2)
    return window_data_train, window_data_test


def model_test(model, trainLoader, testLoader, device):
    window_correct_test = 0
    window_total_test = 0
    window_correct_train = 0
    window_total_train = 0
    model.eval()
    for i, sample in enumerate(trainLoader, 0):
        labels, data = sample['label'].to(device).to(torch.int64), sample['data'].to(device).float()

        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        window_total_train += labels.size(0)

        window_correct_train += (predicted == labels).sum().item()
    for i, sample_test in enumerate(testLoader, 0):
        labels_test, data_test = sample_test['label'].to(device).to(torch.int64), sample_test['data'].to(device).float()

        outputs = model(data_test)
        _, predicted_test = torch.max(outputs.data, 1)
        window_total_test += labels_test.size(0)
        window_correct_test += (predicted_test == labels_test).sum().item()

    correct_percentage_train = window_correct_train / window_total_train
    correct_percentage_test = window_correct_test / window_total_test

    return correct_percentage_train, correct_percentage_test


def get_one_axis(window_data, axis='v'):
    if axis == 'v':
        window_data = window_data[:, 0, :]
    elif axis == 'm':
        window_data = window_data[:, 1, :]
    elif axis == 'a':
        window_data = window_data[:, 2, :]
    else:
        raise Exception('wrong axis')
    window_data = window_data[:, np.newaxis, :]
    return window_data


def CNN_debug(epochs=10, window_step=1, window_size=48, dataset=all_group_number, model_save=False,
              TRAIN_JUST_ONE=False, GET_OUTPUT_PROB=False):
    # init
    currentDateAndTime = datetime.now()
    currentTime = currentDateAndTime.strftime("%m_%d_%H_%M_%S")
    save_dir = os.path.join('visualize/CNN_acc_loss', ('30P' if dataset == all_group_number_30 else '12P') + currentTime)
    os.makedirs(save_dir)
    dataset = dataset
    EPOCH = epochs
    WINDOW_STEP = window_step
    WINDOW_SIZE = window_size
    correct_person_count = 0
    correct_person_list = []

    # making dataset
    total_person_count = len(dataset)
    if dataset == all_group_number_30:
        paitent_makers, window_labels, window_data = window_oper(dataset, WINDOW_SIZE, WINDOW_STEP, dataset='30')
    elif dataset == all_group_number:
        paitent_makers, window_labels, window_data = window_oper(dataset, WINDOW_SIZE, WINDOW_STEP, dataset='12')
    else:
        raise Exception('wrong dataset')
    window_labels = np.array(window_labels)
    window_data = np.array(window_data)

    # get one axis
    window_data = get_one_axis(window_data, axis='a')
    print('current: EPOCH %d W_SIZE %d W_STEP %d' % (EPOCH, WINDOW_SIZE, WINDOW_STEP))

    if TRAIN_JUST_ONE:
        dataset = [dataset[0]]

    for number in dataset:
        # divide window for train and test
        testing_idx = [i for i, x in enumerate(paitent_makers) if x == number]
        training_idx = [i for i, x in enumerate(paitent_makers) if x != number]

        testing_labels = window_labels[testing_idx]
        testing_data = window_data[testing_idx, :]
        training_labels = window_labels[training_idx]
        training_data = window_data[training_idx, :]
        # normalize based on the max an min of training data
        # training_data, testing_data = normalize(training_data, testing_data)

        # making trainset and testset and loader
        trainset = DMDDataset(training_labels, training_data, dimension=1)  # if 2d conv, dimension =2
        testset = DMDDataset(testing_labels, testing_data, dimension=1)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                                  shuffle=True)

        testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                                 shuffle=False)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # net = CNN_DMD(WINDOW_SIZE).float().to(device)
        # net = CNN_var(WINDOW_SIZE).float().to(device)
        # net = CNN_Pooling(WINDOW_SIZE, N_OF_Module=2).float().to(device)
        net = one_axis_CNN(WINDOW_SIZE, N_OF_Module=2).float().to(device)
        optimizer = optim.Adam(net.parameters())
        loss_function = nn.CrossEntropyLoss()

        # acc epoch init and acc before training
        test_acc_epoch = []
        train_acc_epoch = []
        loss_epoch = []
        correct_percentage_train, correct_percentage_test = model_test(net, trainLoader=trainloader,
                                                                       testLoader=testloader, device=device)
        print('%s: epoch %d train per:%.3f,test per:%.3f,run loss %.3f' % (
            number, 0, correct_percentage_train, correct_percentage_test, 0))
        train_acc_epoch.append(correct_percentage_train)
        test_acc_epoch.append(correct_percentage_test)
        loss_epoch.append(0)
        # print('patient: ', number, ' train start')
        for epochs in range(EPOCH):
            running_loss = 0
            net.train()
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
            correct_percentage_train, correct_percentage_test = model_test(net, trainLoader=trainloader,
                                                                           testLoader=testloader, device=device)

            if epochs == EPOCH - 1:
                if correct_percentage_test > 0.5:
                    correct_person_count += 1
                    correct_person_list.append(number)

            print('%s: epoch %d train per:%.3f,test per:%.3f,run loss %.3f' % (
                number, epochs + 1, correct_percentage_train, correct_percentage_test, running_loss))
            train_acc_epoch.append(correct_percentage_train)
            test_acc_epoch.append(correct_percentage_test)
            loss_epoch.append(running_loss)

            if TRAIN_JUST_ONE and GET_OUTPUT_PROB:
                sample = trainloader[0]
                labels, data = sample['label'].to(device).to(torch.int64), sample['data'].to(device).float()
                net.eval()
                outputs = net(data)
                print(outputs)
        # plotting
        plt.figure(figsize=(18, 15))
        plt.ylabel('Magnitude on ', fontsize=20)
        ax = plt.subplot(311)
        ax.plot(range(EPOCH + 1), train_acc_epoch, label='train_acc_epoch')
        ax.set_xlabel('epoch')
        ax.set_ylabel('train_acc_epoch')
        ax.grid()
        ax.legend()

        ax = plt.subplot(312)
        ax.plot(range(EPOCH + 1), test_acc_epoch, label='test_acc_epoch')
        ax.set_xlabel('epoch')
        ax.set_ylabel('test_acc_epoch')
        ax.grid()
        ax.legend()

        ax = plt.subplot(313)
        ax.plot(range(EPOCH + 1), loss_epoch, label='loss')
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss_epoch')
        ax.grid()
        ax.legend()

        plt.savefig(os.path.join(save_dir, number))
    print('images in %s' % save_dir)
    print('total_person_count acc: %.3f' % (correct_person_count / total_person_count))
    print(correct_person_list)


CNN_debug()

# j = 0
#
# for EPOCH in [100, 500, 1000, 5000, 10000, 50000]:
#     for WINDOW_SIZE in [30, 33, 90, 100]:
#         for WINDOW_STEP in [WINDOW_SIZE, int(WINDOW_SIZE / 2), int(WINDOW_SIZE / 3)]:
#             correct = 0
#             total = len(all_group_number)
#             paitent_makers, window_labels, window_data = window_oper(all_group_number, WINDOW_SIZE, WINDOW_STEP)
#             window_labels = np.array(window_labels)
#             window_data = np.array(window_data)
#             print('current: EPOCH %d W_SIZE %d W_STEP %d' % (EPOCH, WINDOW_SIZE, WINDOW_STEP))
#             for number in all_group_number:
#                 j += 1
#                 testing_idx = [i for i, x in enumerate(paitent_makers) if x == number]
#                 training_idx = [i for i, x in enumerate(paitent_makers) if x != number]
#
#                 testing_labels = window_labels[testing_idx]
#                 testing_data = window_data[testing_idx, :]
#                 training_labels = window_labels[training_idx]
#                 training_data = window_data[training_idx, :]
#
#                 trainset = DMDDataset(training_labels, training_data)
#                 testset = DMDDataset(testing_labels, testing_data)
#
#                 trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
#                                                           shuffle=True)
#                 testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
#                                                          shuffle=False)
#
#                 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#                 net = CNN_DMD(WINDOW_SIZE).float().to(device)
#                 optimizer = optim.Adam(net.parameters())
#                 loss_function = nn.CrossEntropyLoss()
#                 # print('patient: ', number, ' train start')
#                 for epoch in range(EPOCH):
#                     running_loss = 0
#                     for i, (sample) in enumerate(trainloader, 0):
#                         # get the inputs
#
#                         labels, data = sample['label'].to(device).to(torch.int64), sample['data'].to(device).float()
#
#                         optimizer.zero_grad()
#                         outputs = net(data)
#
#                         loss = loss_function(outputs, labels)
#                         loss.backward()
#                         optimizer.step()
#
#                         running_loss += loss.item()
#                     # print(running_loss)
#                 window_correct = 0
#                 window_total = 0
#                 # print('patient: ', number, ' train finished')
#                 with torch.no_grad():
#                     for i, sample in enumerate(testloader, 0):
#                         labels, data = sample['label'].to(device).to(torch.int64), sample['data'].to(device).float()
#
#                         outputs = net(data)
#                         _, predicted = torch.max(outputs.data, 1)
#                         window_total += labels.size(0)
#
#                         window_correct += (predicted == labels).sum().item()
#                         # print(predicted)
#                         # print(labels)
#                         # print(window_correct)
#                         # print(window_total)
#                 correct_percentage = window_correct / window_total
#
#                 if correct_percentage > 0.5:
#                     correct += 1
#                     # print(' patient: %s predict correct, percentage:%.2f,loss:%.5f' % (number,  correct_percentage, running_loss))
#                 # else:
#                 # print(' patient: %s predict wrong, percentage:%.2f,loss:%.5f' % (number,  correct_percentage, running_loss))
#
#                 # print('total correct percentage:', correct / total)
#             csv_writer(EPOCH, WINDOW_STEP, WINDOW_SIZE, 'bs_' + str(BATCH_SIZE) + '_lr_' + str(LEARN_RATE),
#                        (correct / total))
