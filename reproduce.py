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
from result_record import csv_writer
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
LEARN_RATE = 1e-3
BATCH_SIZE = 2048
# EPOCH = 100000
NUM_WORKERS = 0

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


# np.random.seed(1)

def normalize(window_data_train, window_data_test):
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
        # labels_test = (labels_test + 1) % 2 #SHAHBAZ
        # print(labels_test)

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

def save_img_epoch_acc(train_acc_epoch,test_acc_epoch,loss_epoch,EPOCH,save_dir,number):
    # plotting
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
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, number))


def save_img_prob_epoch(epochs,output_values,save_dir,number):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # np.savetxt(os.path.join(save_dir, str(fold_i + 1) + '_' + str(epochs + 1)),output_values,delimiter=',')
    plt.figure(figsize=(18, 15))
    plt.scatter(range(output_values.shape[0]), output_values[:, 0], linestyle='solid', label='TD')
    plt.scatter(range(output_values.shape[0]), output_values[:, 1], linestyle='solid', label='DMD')
    plt.legend()
    plt.savefig(os.path.join(save_dir, number + '_' + str(epochs + 1)))


def one_fold_training(number, patient_makers, window_labels, window_data, NORMALIZE, EPOCH, WINDOW_SIZE, device,
                      GET_OUTPUT_PROB, save_dir, SAVE_IMAGE, ):
    # divide window for train and test
    # print(number + ' is on ' + str(device))
    start_time = time.time()
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
        training_data, testing_data = normalize(training_data, testing_data)

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
    net = ResNet1D(in_channels=3, base_filters=16, kernel_size=3, stride=2, groups=1, n_block=24, n_classes=2,
                   downsample_gap=6, increasefilter_gap=6, use_bn=True, use_do=True, verbose=False)
    # net = CNN_Pooling(WINDOW_SIZE, N_OF_Module=2).float().to(device)
    net = nn.DataParallel(net).to(device)
    # net = one_axis_CNN(WINDOW_SIZE, N_OF_Module=2).float().to(device)
    # net = CNN_for_window_FFT(WINDOW_SIZE, N_OF_Module=2).float().to(device)
    optimizer = optim.Adam(net.parameters(), lr=LEARN_RATE)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.2)
    scheduler = MultiStepLR(optimizer, milestones=[10, 50], gamma=0.2)
    loss_function = nn.CrossEntropyLoss()

    # acc epoch init and acc before training
    test_acc_epoch = []
    train_acc_epoch = []
    loss_epoch = []
    correct_percentage_test = 0

    # correct_percentage_train, correct_percentage_test, output_values, conf_total_test = model_test(net,
    #                                                                                                trainLoader=trainloader,
    #                                                                                                testLoader=testloader,
    #                                                                                                device=device,
    #                                                                                                GET_OUTPUT_PROB=GET_OUTPUT_PROB)
    #
    # print('%s: epoch %d train per:%.3f,test per:%.3f,run loss %.3f lr for next epoch %f conf %.3f' % (
    #     number, 0, correct_percentage_train, correct_percentage_test, 0, scheduler.get_last_lr()[0], conf_total_test))

    # train_acc_epoch.append(correct_percentage_train)
    # test_acc_epoch.append(correct_percentage_test)
    # loss_epoch.append(0)

    # print('patient: ', number, ' train start')
    # with torch.profiler.profile(
    #         activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
    # print('train start %.3f for preparing training set length %d' % ((time.time()-start_time), len(trainset)))
    start_time = time.time()
    for epochs in range(EPOCH):

        running_loss = 0
        net.train()
        # with tqdm(trainloader) as tepoch:
        # torch.cuda.synchronize()

        for i, (sample) in enumerate(trainloader, 0):
            # get the inputs
            labels, data = sample['label'].to(device).to(torch.int64), sample['data'].to(device).float()

            optimizer.zero_grad()
            outputs = net(data)

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # running_loss +=

        # StepLR step
        scheduler.step()

        if GET_OUTPUT_PROB:
            save_img_prob_epoch(epochs, output_values, save_dir, number)
        # print(time.time() - start_time)
        if epochs == EPOCH - 1:
            correct_percentage_train, correct_percentage_test, output_values, conf_total_test = model_test(net,
                                                                                                           trainLoader=trainloader,
                                                                                                           testLoader=testloader,
                                                                                                           device=device,
                                                                                                           GET_OUTPUT_PROB=GET_OUTPUT_PROB)
            # torch.cuda.synchronize()
            print(
                '%s: epoch %d train per:%.3f,test per:%.3f,loss: %.3f, lr for next epoch %f, time costing %.3f, conf %.3f'
                % (number, epochs + 1, correct_percentage_train, correct_percentage_test, loss.item(),
                   scheduler.get_last_lr()[0], time.time() - start_time, conf_total_test))

    # train_acc_epoch.append(correct_percentage_train)
    # test_acc_epoch.append(correct_percentage_test)
    # loss_epoch.append(running_loss)
    # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

    if SAVE_IMAGE:
        save_img_epoch_acc(train_acc_epoch, test_acc_epoch, loss_epoch, EPOCH, save_dir, number)

    return correct_percentage_test



def CNN_debug(epochs=20, window_step=40, window_size=80, people_number=all_group_number_30,
              dataset_path=L3_path_30, model_save=False, NORMALIZE=False,
              TRAIN_JUST_ONE=False, GET_OUTPUT_PROB=False, SAVE_IMAGE=True, BAD_SAMPLE_INVESTIGATE=False,
              BAD_SAMPLE_KICKOUT=False, ZERO_OUT=True, zero_out_freq=10):

    # init
    currentDateAndTime = datetime.now()
    currentTime = currentDateAndTime.strftime("%m_%d_%H_%M_%S")
    save_dir = os.path.join('visualize/CNN_acc_loss',
                            ('30P' if people_number == all_group_number_30 else '12P') + currentTime)

    people_number = people_number

    # if torch.cuda.device_count() > 1:
    #     print('%d GPUs are using' % torch.cuda.device_count())
    EPOCH = epochs

    WINDOW_STEP = window_step  # 40
    WINDOW_SIZE = window_size

    if people_number == all_group_number_30:
        # patient_makers, window_labels, window_data = window_oper(dataset, WINDOW_SIZE, WINDOW_STEP, people_number='30')
        patient_makers, window_labels, window_data = window_oper(people_number, WINDOW_SIZE, WINDOW_STEP,
                                                                 people_number='30', dataset_path=dataset_path, )
    elif people_number == all_group_number:
        # patient_makers, window_labels, window_data = window_oper_HS_3windows(dataset, WINDOW_SIZE, WINDOW_STEP,
        # dataset='12')
        # patient_makers, window_labels, window_data = window_oper(dataset, WINDOW_SIZE, WINDOW_STEP, people_number='12',
        #                                                          zero_out=True, zero_out_freq=10)
        # patient_makers, window_labels, window_data = window_FFT_oper(dataset, WINDOW_SIZE, WINDOW_STEP, people_number='12')
        patient_makers, window_labels, window_data = window_oper(people_number, WINDOW_SIZE, WINDOW_STEP,
                                                                 people_number='12', dataset_path=dataset_path, )
    else:
        raise Exception('wrong dataset')
    window_labels = np.array(window_labels)
    window_data = np.array(window_data)
    # print(window_labels.shape)
    # print(window_data.shape)
    # get one axis
    # window_data = get_one_axis(window_data, axis='a')
    # print('current: EPOCH %d W_SIZE %d W_STEP %d' % (EPOCH, WINDOW_SIZE, WINDOW_STEP))
    people_number = [i.split('.')[0] for i in os.listdir(dataset_path)]
    if TRAIN_JUST_ONE:
        # dataset = [dataset[0]]

        people_number = ['990023015']

    # if BAD_SAMPLE_INVESTIGATE:
    #     people_number = bad_sample

    correct_person_count = 0
    correct_person_list = []
    wrong_person_list = []
    correct_percentages = []
    person_acc_lists = []
    # making dataset
    # people_number = ['23015']
    total_person_count = len(people_number)
    # pool = Pool(1)
    parameters_lists = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    for fold_i, number in enumerate(people_number):
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
        value = one_fold_training(number, patient_makers, window_labels, window_data,
                                  NORMALIZE, EPOCH, WINDOW_SIZE, device,
                                  GET_OUTPUT_PROB, save_dir, SAVE_IMAGE)
        correct_percentages.append(value)
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
    correct_person_count = np.sum(correct_percentages > 0.5)
    for i in iter(np.where(correct_percentages > 0.5)[0].tolist()):
        correct_person_list.append(people_number[i])
    for i in iter(np.where(correct_percentages <= 0.5)[0].tolist()):
        wrong_person_list.append(people_number[i])
    if SAVE_IMAGE:
        print('images in %s' % save_dir)
    print('total_person_count acc: %.3f' % (correct_person_count / total_person_count))
    # print('correct person list:', correct_person_list)
    print('wrong person list:', wrong_person_list)
    return (correct_person_count / total_person_count), wrong_person_list


def multiple_running(repeat_time=1, save_dir='./save_result', word_marker='6_min_walk_zeroout_30',
                     dataset_path=six_min_path_30, NORMALIZE=False,
                     WINDOW_SIZE=80, WINDOW_STEP=5, people_number=all_group_number_30, EPOCHS=100):
    acc_s = []
    wrong_lists = []
    start = time.time()

    for i in range(repeat_time):
        # print('multiple_running for ' + str(repeat_time))
        print('current: ' + word_marker)
        acc, wrong_list = CNN_debug(epochs=EPOCHS, window_step=WINDOW_STEP, window_size=WINDOW_SIZE,
                                    people_number=people_number,
                                    dataset_path=dataset_path, model_save=False, NORMALIZE=NORMALIZE,
                                    TRAIN_JUST_ONE=False, GET_OUTPUT_PROB=False, SAVE_IMAGE=False,
                                    BAD_SAMPLE_INVESTIGATE=False, BAD_SAMPLE_KICKOUT=False,
                                    ZERO_OUT=False, zero_out_freq=6)

        acc_s.append(acc)
        wrong_lists += wrong_list
        print('it took %.2f seconds for a whole test' % (time.time() - start))
    avg_acc = np.mean(np.array(acc_s))
    max_acc = np.max(np.array(acc_s))
    min_acc = np.min(np.array(acc_s))
    variance = np.var(np.array(acc_s))
    wrong_count = pd.value_counts(wrong_lists)
    word_marker = word_marker
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    currentTime = datetime.now().strftime("%m_%d_%H_%M_%S")
    dataframe = pd.DataFrame({'avg_acc': [avg_acc], 'max_acc': [max_acc], 'min_acc': [min_acc], 'variance': [variance]})
    dataframe.to_csv(os.path.join(save_dir, word_marker + '_' + currentTime + 'acc.csv'), index=True, sep=',')
    dataframe = pd.DataFrame([wrong_count])
    dataframe.to_csv(os.path.join(save_dir, word_marker + '_' + currentTime + 'wrong_list.csv'), index=True, sep=',')


# multiple_running(repeat_time=1, save_dir='./save_result', word_marker='albara_model_way', people_number=all_group_number,
#                      dataset_path='dataset/downsample', NORMALIZE=False,
#                      WINDOW_SIZE=30, WINDOW_STEP=30)

if __name__ == "__main__":
    # multi running to get limit
    w_size = 160
    w_step = 5
    NORMALIZE = False
    print('resnet_6_min_walk and resnet_100_meter_walk')
    torch.multiprocessing.set_start_method('spawn')

    multiple_running(repeat_time=1, save_dir='./save_result', NORMALIZE=NORMALIZE,
                     word_marker='resnet_100_meter_walk' + str(w_size) + '_step' + str(
                         w_step) + '_Norm_' + str(NORMALIZE),
                     dataset_path='dataset/30_dmd_data_set/100-meter-walk',
                     WINDOW_SIZE=w_size, WINDOW_STEP=w_step)

    multiple_running(repeat_time=1, save_dir='./save_result', NORMALIZE=NORMALIZE,
                     word_marker='resnet_6_min_walk' + str(w_size) + '_step' + str(
                         w_step) + '_Norm_' + str(NORMALIZE),
                     dataset_path='dataset/30_dmd_data_set/6-min-walk',
                     WINDOW_SIZE=w_size, WINDOW_STEP=w_step)

# for Norm in [False]:
#     for window_size in [160]:
#         for window_step in [int(window_size/2)]:
# multiple_running(repeat_time=1, save_dir='./save_result', NORMALIZE=Norm,
#                  word_marker='100_meter_walk_raw_size' + str(window_size) + '_step' + str(
#                      window_step),
#                  dataset_path='dataset/30_dmd_data_set/100-meter-walk',
#                  WINDOW_SIZE=window_size, WINDOW_STEP=window_step)
#
# multiple_running(repeat_time=1, save_dir='./save_result', NORMALIZE=Norm,
#                  word_marker='100_meter_walk_zeroout_15size' + str(window_size) + '_step' + str(
#                      window_step),
#                  dataset_path='dataset/ZeroHighFreq/hundred_meter_26people_freq_15',
#                  WINDOW_SIZE=window_size, WINDOW_STEP=window_step)
#
# multiple_running(repeat_time=1, save_dir='./save_result', NORMALIZE=Norm,
#                  word_marker='100_meter_walk_zeroout_30size' + str(window_size) + '_step' + str(
#                      window_step) + '_Norm_' + str(Norm),
#                  dataset_path='dataset/ZeroHighFreq/hundred_meter_26people_freq_30',
#                  WINDOW_SIZE=window_size, WINDOW_STEP=window_step)
# multiple_running(repeat_time=1, save_dir='./save_result', NORMALIZE=Norm,
#                  word_marker='6_min_walk_raw_size' + str(window_size) + '_step' + str(
#                      window_step),
#                  dataset_path='dataset/30_dmd_data_set/6-min-walk',
#                  WINDOW_SIZE=window_size, WINDOW_STEP=window_step)
#
# multiple_running(repeat_time=1, save_dir='./save_result', NORMALIZE=Norm,
#                  word_marker='6_min_walk_zeroout_15size' + str(window_size) + '_step' + str(
#                      window_step),
#                  dataset_path='dataset/ZeroHighFreq/six_min_30people_freq_15',
#                  WINDOW_SIZE=window_size, WINDOW_STEP=window_step)
#
# multiple_running(repeat_time=1, save_dir='./save_result', NORMALIZE=Norm,
#                  word_marker='6_min_walk_zeroout_30size' + str(window_size) + '_step' + str(
#                      window_step) + '_Norm_' + str(Norm),
#                  dataset_path='dataset/ZeroHighFreq/six_min_30people_freq_30',
#                  WINDOW_SIZE=window_size, WINDOW_STEP=window_step)


# multiple_running(repeat_time=10, save_dir='./save_result', NORMALIZE=True,
#                  word_marker='TEN_TIMES_six_min_30people_raw_size' + str(160) + '_step' + str(
#                      10) + '_Norm_' + str(True),
#                  dataset_path='dataset/30_dmd_data_set/6-min-walk',
#                  WINDOW_SIZE=160, WINDOW_STEP=10)
#
# multiple_running(repeat_time=10, save_dir='./save_result', NORMALIZE=True,
#                  word_marker='TEN_TIMES_100_meter_walk_raw_size' + str(160) + '_step' + str(
#                      5),
#                  dataset_path='dataset/30_dmd_data_set/100-meter-walk',
#                  WINDOW_SIZE=160, WINDOW_STEP=5)

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
