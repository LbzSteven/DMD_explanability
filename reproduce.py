import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm
from window import window_oper, window_oper_HS_3windows, window_FFT_oper
from CNN_torch import CNN_DMD, CNN_var, CNN_Pooling, one_axis_CNN, CNN_for_window_FFT
from resnet1d import ResNet1D

from DMDdataset import DMDDataset
from result_record import csv_writer_acc_count_wpl
# import torchvision.transforms as transforms
from torch.multiprocessing import Pool
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torch.nn.functional as F
from datetime import datetime
from constants import DMD_group_number, TD_group_number, all_group_number, low_sample_rate, high_sample_rate, \
    TD_group_number_30, DMD_group_number_30, all_group_number_30, six_min_path_30, hundred_meter_path_26, L3_path_30, \
    exclude_list_100m
import pandas as pd

# WINDOW_SIZE = 33
# WINDOW_STEP = 33

NUM_WORKERS = 0


# np.random.seed(1)

def normalize_oper(window_data_train, window_data_test):
    if window_data_train.ndim == 2:
        t_train = window_data_train[:, np.newaxis]
        t_test = window_data_test[:, np.newaxis]
    else:
        t_train = np.moveaxis(window_data_train, 1, 2)
        t_test = np.moveaxis(window_data_test, 1, 2)
    if window_data_train.shape[2] == 1:
        max_3_axis = np.max(t_train.reshape(-1, window_data_train.shape[2]))
        min_3_axis = np.min(t_train.reshape(-1, window_data_train.shape[2]))
    else:
        max_3_axis = np.max(t_train.reshape(-1, window_data_train.shape[2]), axis=0)
        min_3_axis = np.min(t_train.reshape(-1, window_data_train.shape[2]), axis=0)
    for i in range(window_data_train.shape[1]):
        t_train[:, :, i] = (t_train[:, :, i] - min_3_axis[i]) / (max_3_axis[i] - min_3_axis[i])
        t_test[:, :, i] = (t_test[:, :, i] - min_3_axis[i]) / (max_3_axis[i] - min_3_axis[i])
        # t_train[:, :, i] = (t_train[:, :, i] + 3) / (3 + 3)
    window_data_train = np.moveaxis(t_train, 1, 2)
    window_data_test = np.moveaxis(t_test, 1, 2)
    return window_data_train, window_data_test


def model_test(model, trainLoader, testLoader, device, GET_OUTPUT_PROB=False):
    window_correct_test = 0
    window_total_test = 0
    window_correct_train = 0
    window_total_train = 0
    conf_total_test = 0
    conf_total_train = 0
    how_many_batch = 0
    model.eval()
    output_value = []
    output_values = None
    for i, sample in enumerate(trainLoader, 0):
        labels, data = sample['label'].to(device).to(torch.int64), sample['data'].to(device).float()

        outputs = model(data)
        # conf_total_train += torch.mean(outputs[:, labels[]])#SHAHBAZ
        _, predicted = torch.max(outputs.data, 1)
        window_total_train += labels.size(0)

        window_correct_train += (predicted == labels).sum().item()
    for i, sample_test in enumerate(testLoader, 0):
        labels_test, data_test = sample_test['label'].to(device).to(torch.int64), sample_test['data'].to(device).float()
        outputs = model(data_test)
        how_many_batch += 1
        conf_total_test += torch.mean(outputs[:, labels_test[0]])  # SHAHBAZ
        # print(outputs)
        _, predicted_test = torch.max(outputs.data, 1)
        window_total_test += labels_test.size(0)
        window_correct_test += (predicted_test == labels_test).sum().item()
        if GET_OUTPUT_PROB:
            output_value.append(outputs.cpu().detach().numpy())
    if GET_OUTPUT_PROB:
        output_values = np.concatenate(output_value)
    correct_percentage_train = window_correct_train / window_total_train
    correct_percentage_test = window_correct_test / window_total_test
    conf_total_test = conf_total_test / how_many_batch

    return correct_percentage_train, correct_percentage_test, output_values, conf_total_test


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


def save_img_epoch_acc(train_acc_epoch, test_acc_epoch, loss_epoch, EPOCH, save_dir, number):
    # plotting
    save_dir = os.path.join(save_dir, 'epoch_acc')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.figure(figsize=(18, 15))
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
    plt.close()


def save_img_prob_epoch(epochs, output_values, save_dir, number):
    # np.savetxt(os.path.join(save_dir, str(fold_i + 1) + '_' + str(epochs + 1)),output_values,delimiter=',')
    plt.figure(figsize=(18, 15))
    plt.scatter(range(output_values.shape[0]), output_values[:, 0], linestyle='solid', label='TD')
    plt.scatter(range(output_values.shape[0]), output_values[:, 1], linestyle='solid', label='DMD')
    plt.legend()
    plt.savefig(os.path.join(save_dir, number + '_' + str(epochs + 1)))
    plt.close()


def one_fold_training(model, arg_list, number, patient_makers, window_labels, window_data, NORMALIZE, EPOCH, device,
                      GET_OUTPUT_PROB, save_dir, SAVE_IMAGE, LEARN_RATE, BATCH_SIZE):
    # divide window for train and test
    # print(number + ' is on ' + str(device))
    # start_time = time.time()

    if (GET_OUTPUT_PROB or SAVE_IMAGE) and (not os.path.exists(save_dir)):
        os.makedirs(save_dir)

    testing_idx = [i for i, x in enumerate(patient_makers) if x == number]
    training_idx = [i for i, x in enumerate(patient_makers) if x != number]
    # if BAD_SAMPLE_KICKOUT:
    #     training_idx = [i for i in training_idx if i not in bad_sample]
    testing_labels = window_labels[testing_idx]
    testing_data = window_data[testing_idx, :]
    training_labels = window_labels[training_idx]
    training_data = window_data[training_idx, :]

    # normalize based on the max an min of training data
    if NORMALIZE:
        training_data, testing_data = normalize_oper(training_data, testing_data)

    # making trainset and testset and loader
    trainset = DMDDataset(training_labels, training_data, dimension=1)  # if 2d conv, dimension =2
    testset = DMDDataset(testing_labels, testing_data, dimension=1)
    # trainset = DMDDataset(training_labels, training_data)  # if 2d conv, dimension =2
    # testset = DMDDataset(testing_labels, testing_data)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                              shuffle=True, pin_memory=False)

    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                             shuffle=False, pin_memory=False)

    # net = CNN_DMD(WINDOW_SIZE).float().to(device)
    # net = CNN_var(WINDOW_SIZE).float().to(device)
    # net = ResNet1D(in_channels=3, base_filters=16, kernel_size=3, stride=2, groups=1, n_block=12, n_classes=2,
    #                downsample_gap=12, increasefilter_gap=12, use_bn=True, use_do=True, verbose=False)
    # net = CNN_Pooling(WINDOW_SIZE, N_OF_Module=2).float().to(device)
    # net = nn.DataParallel(net).to(device)
    # net = one_axis_CNN(WINDOW_SIZE, N_OF_Module=2).float().to(device)
    # net = CNN_for_window_FFT(WINDOW_SIZE, N_OF_Module=2).float().to(device)
    net = model(*arg_list)
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=LEARN_RATE)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.2)
    scheduler = MultiStepLR(optimizer, milestones=[10, 50], gamma=0.2)
    loss_function = nn.CrossEntropyLoss()

    # acc epoch init and acc before training
    test_acc_epoch = []
    train_acc_epoch = []
    loss_epoch = []
    correct_percentage_test = 0
    output_values_lists = []
    if SAVE_IMAGE:
        correct_percentage_train, correct_percentage_test, output_values, conf_total_test = model_test(net,
                                                                                                       trainLoader=trainloader,
                                                                                                       testLoader=testloader,
                                                                                                       device=device,
                                                                                                       GET_OUTPUT_PROB=GET_OUTPUT_PROB)
        train_acc_epoch.append(correct_percentage_train)
        test_acc_epoch.append(correct_percentage_test)
        loss_epoch.append(0)
    # print('%s: epoch %d train per:%.3f,test per:%.3f,run loss %.3f lr for next epoch %f conf %.3f' % (
    #     number, 0, correct_percentage_train, correct_percentage_test, 0, scheduler.get_last_lr()[0], conf_total_test))

    # profiler
    # with torch.profiler.profile(
    #         activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
    # print('train start %.3f for preparing training set length %d' % ((time.time()-start_time), len(trainset)))
    start_time = time.time()
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

        # StepLR step
        scheduler.step()

        # print(time.time() - start_time)
        if (epochs == EPOCH - 1) or SAVE_IMAGE:
            correct_percentage_train, correct_percentage_test, output_values, conf_total_test = model_test(net,
                                                                                                           trainLoader=trainloader,
                                                                                                           testLoader=testloader,
                                                                                                           device=device,
                                                                                                           GET_OUTPUT_PROB=GET_OUTPUT_PROB)
            if GET_OUTPUT_PROB:
                output_values_lists.append(output_values)
                # save_img_prob_epoch(epochs, output_values, save_dir, number)
            train_acc_epoch.append(correct_percentage_train)
            test_acc_epoch.append(correct_percentage_test)
            loss_epoch.append(running_loss)
            # print(
            #     '%s: epoch %d train per:%.3f,test per:%.3f,loss: %.3f, lr for next epoch %f, time costing %.3f, conf %.3f'
            #     % (number, epochs + 1, correct_percentage_train, correct_percentage_test, running_loss,
            #        scheduler.get_last_lr()[0], time.time() - start_time, conf_total_test))

    # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

    if SAVE_IMAGE:
        save_dir = os.path.join(save_dir, 'epoch_acc')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(os.path.join(save_dir, number + '.npy'), output_values_lists)
        save_img_epoch_acc(train_acc_epoch, test_acc_epoch, loss_epoch, EPOCH, save_dir, number)

    if GET_OUTPUT_PROB:
        save_dir = os.path.join(save_dir, 'origin_output_prob')
        output_values_lists = np.array(output_values_lists)
        # print('output_values_lists shape: ', output_values_lists.shape)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(os.path.join(save_dir, number + '.npy'), output_values_lists)
    return correct_percentage_test


def n_fold_exper(model, arg_list, word_marker, lr, batch_size, epochs, window_step, window_size,
                 dataset_path, normalize, num_of_gpu, get_output_prob, save_image, TRAIN_JUST_ONE,
                 BAD_SAMPLE_KICKOUT, ZERO_OUT, zero_out_freq, model_save, BAD_SAMPLE_INVESTIGATE, dataset_marker='30'):
    # init
    start_time = time.time()
    currentDateAndTime = datetime.now()
    currentTime = currentDateAndTime.strftime("%m_%d_%H_%M")
    save_dir = os.path.join('save_result/', word_marker + '_Epoch_acc_prob_out_' + currentTime)

    EPOCH = epochs
    WINDOW_STEP = window_step
    WINDOW_SIZE = window_size

    patient_makers, window_labels, window_data = window_oper(WINDOW_SIZE, WINDOW_STEP,
                                                             dataset_marker=dataset_marker, dataset_path=dataset_path, )

    # patient_makers, window_labels, window_data = window_oper_HS_3windows(dataset, WINDOW_SIZE, WINDOW_STEP,dataset='12')
    # patient_makers, window_labels, window_data = window_FFT_oper(dataset, WINDOW_SIZE, WINDOW_STEP, people_number='12')

    # get one axis
    # window_data = get_one_axis(window_data, axis='a')

    people_number = [i.split('.')[0] for i in os.listdir(dataset_path)]  # regenerate correct people number
    if TRAIN_JUST_ONE:
        people_number = [people_number[0]]

    # if BAD_SAMPLE_INVESTIGATE:
    #     people_number = bad_sample

    correct_person_count = 0
    correct_person_list = []
    wrong_person_list = []
    correct_percentages = []
    person_acc_lists = []

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_person_count = len(people_number)
    NUM_OF_POOL = math.ceil(len(people_number) / num_of_gpu)
    fold_i = 0
    for i_pool in range(NUM_OF_POOL):
        # parameters_lists = []
        pool = Pool(num_of_gpu)
        parameters_lists = []
        for i_process in range(num_of_gpu):
            number = people_number[fold_i]
            device = 'cuda:' + str(fold_i % num_of_gpu)
            parameters_lists.append([model, arg_list, number, patient_makers, window_labels, window_data,
                                     normalize, EPOCH, device, get_output_prob,
                                     save_dir, save_image, lr, batch_size])
            fold_i += 1
            if fold_i == len(people_number):
                break
        iter_para = iter(parameters_lists)
        pool_mapping_results = pool.starmap_async(one_fold_training, iter_para)
        pool.close()
        pool.join()
        torch.cuda.empty_cache()
        for value in pool_mapping_results.get():
            correct_percentages.append(value)

    # dummy iteration method
    # for fold_i, number in enumerate(people_number):
    #
    #
    #     value = one_fold_training(number, patient_makers, window_labels, window_data,
    #                               NORMALIZE, EPOCH, WINDOW_SIZE, device,
    #                               GET_OUTPUT_PROB, save_dir, SAVE_IMAGE,LEARN_RATE)
    #     correct_percentages.append(value)

    # person variance
    # person_acc_list = []
    # print(number + ' will be on cuda:' + str(fold_i % 6))
    # device = 'cuda:' + str(fold_i % 6)
    # for i in range(10):
    #     parameters_lists.append([number, patient_makers, window_labels, window_data,
    #                              NORMALIZE, EPOCH, WINDOW_SIZE, device,
    #                              GET_OUTPUT_PROB, save_dir, SAVE_IMAGE])
    # parameters_lists.append([number, patient_makers, window_labels, window_data,
    #                          NORMALIZE, EPOCH, WINDOW_SIZE, device,
    #                          GET_OUTPUT_PROB, save_dir, SAVE_IMAGE])
    # iter_para = iter(parameters_lists)
    # pool_mapping_results = pool.starmap_async(one_fold_training, iter_para)
    # pool.close()
    # pool.join()
    # for value in pool_mapping_results.get():
    # person_acc_list.append(value)
    # person_acc_lists.append(person_acc_list)
    # correct_percentages.append(value)
    # print(person_acc_lists)
    # person_acc_lists = np.array(person_acc_lists)
    # mean_person = np.mean(person_acc_lists, axis=1)
    # std_person = np.std(person_acc_lists, axis=1)
    # min_person = np.min(person_acc_lists, axis=1)
    # max_person = np.max(person_acc_lists, axis=1)
    # for fold_i, number in enumerate(people_number):
    #     print('%s has mean of %.3f and std of %.3f, min:%.3f,max:%.3f'
    #           % (number, mean_person[fold_i], std_person[fold_i], min_person[fold_i], max_person[fold_i]))
    correct_percentages = np.array(correct_percentages)
    # print(correct_percentages)
    correct_person_count = np.sum(correct_percentages > 0.5)
    for i in iter(np.where(correct_percentages > 0.5)[0].tolist()):
        correct_person_list.append(people_number[i])
    for i in iter(np.where(correct_percentages <= 0.5)[0].tolist()):
        wrong_person_list.append(people_number[i])

    csv_writer_acc_count_wpl('model', people_number, 'correct_person_count', 'wrong_person_list', 'time')
    csv_writer_acc_count_wpl(word_marker + '_' + currentTime, correct_percentages, correct_person_count,
                             wrong_person_list, time.time() - start_time)
    print('total_person_count acc: %.3f' % (correct_person_count / total_person_count))
    # print('correct person list:', correct_person_list)
    print('wrong person list:', wrong_person_list)
    return (correct_person_count / total_person_count), wrong_person_list


def multiple_running(model, arg_list, lr, batch_size, dataset_path,
                     normalize, save_image, get_output_prob, w_size, w_step, epochs, num_of_gpu,
                     repeat_time=1, save_mul=False, save_dir_mul='./save_result_mul',
                     word_marker='HEY! WHERE IS YOUR WORD MARKER?'):
    acc_s = []
    wrong_lists = []
    print(word_marker)
    for i in range(repeat_time):
        acc, wrong_list = n_fold_exper(model, arg_list, word_marker, lr, batch_size, epochs, w_step, w_size,
                                       dataset_path, normalize, num_of_gpu, get_output_prob, save_image,
                                       TRAIN_JUST_ONE=False, BAD_SAMPLE_KICKOUT=False,
                                       ZERO_OUT=False, zero_out_freq=False, model_save=False,
                                       BAD_SAMPLE_INVESTIGATE=False, dataset_marker='30')

        acc_s.append(acc)
        wrong_lists += wrong_list

    if save_mul:
        avg_acc = np.mean(np.array(acc_s))
        max_acc = np.max(np.array(acc_s))
        min_acc = np.min(np.array(acc_s))
        variance = np.var(np.array(acc_s))
        wrong_count = pd.value_counts(wrong_lists)
        word_marker = word_marker
        if not os.path.exists(save_dir_mul):
            os.makedirs(save_dir_mul)
        currentTime = datetime.now().strftime("%m_%d_%H_%M_%S")
        dataframe = pd.DataFrame(
            {'avg_acc': [avg_acc], 'max_acc': [max_acc], 'min_acc': [min_acc], 'variance': [variance]})
        dataframe.to_csv(os.path.join(save_dir_mul, word_marker + '_' + currentTime + 'acc.csv'), index=True, sep=',')
        dataframe = pd.DataFrame([wrong_count])
        dataframe.to_csv(os.path.join(save_dir_mul, word_marker + '_' + currentTime + 'wrong_list.csv'), index=True,
                         sep=',')


if __name__ == "__main__":
    # multi running to get limit
    W_SIZE = 160
    W_STEP = 5

    SAVE_IMAGE = True
    GET_OUTPUT_PROB = True
    EPOCHS = 100
    LEARN_RATE = 0.001
    BATCH_SIZE = 2048
    NUM_OF_GPU = 7
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6'
    # print('resnet_100_meter_walk norm true and 6 min true and false')
    torch.multiprocessing.set_start_method('spawn')

    model = ResNet1D
    in_channels = 3
    base_filters = 16
    kernel_size = 3
    stride = 2
    groups = 1
    n_block = 48
    n_classes = 2
    downsample_gap = 12
    increasefilter_gap = 12
    use_bn = True
    use_do = True
    verbose = False

    arg_list = [in_channels, base_filters, kernel_size, stride, groups, n_block, n_classes, downsample_gap,
                increasefilter_gap, use_bn, use_do, verbose]

    NORMALIZE = True

    multiple_running(model, arg_list, LEARN_RATE, BATCH_SIZE, hundred_meter_path_26,
                     NORMALIZE, SAVE_IMAGE, GET_OUTPUT_PROB, W_SIZE, W_STEP, EPOCHS, NUM_OF_GPU,
                     repeat_time=1, save_mul=False, save_dir_mul='./save_result_mul',
                     word_marker='resnet24_avg_softmax_100_meter_walk' + str(W_SIZE) + '_step' + str(
                         W_STEP) + '_Norm_' + str(NORMALIZE))

    NORMALIZE = False

    multiple_running(model, arg_list, LEARN_RATE, BATCH_SIZE, hundred_meter_path_26,
                     NORMALIZE, SAVE_IMAGE, GET_OUTPUT_PROB, W_SIZE, W_STEP, EPOCHS, NUM_OF_GPU,
                     repeat_time=1, save_mul=False, save_dir_mul='./save_result_mul',
                     word_marker='resnet24_avg_softmax_100_meter_walk' + str(W_SIZE) + '_step' + str(
                         W_STEP) + '_Norm_' + str(NORMALIZE))

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
