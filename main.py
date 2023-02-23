import math
import os
import random
import time
from datetime import datetime
from multiprocessing import Pool

import numpy
import numpy as np
import pandas
import pandas as pd
import torch
from torch import optim as optim, nn
from torch.optim.lr_scheduler import MultiStepLR

from CNN_torch import CNN_DMD, CNN_var, CNN_Pooling, one_axis_CNN, CNN_for_window_FFT
from DMDdataset import DMDDataset
from people_characteristic_builder import people_list
from resnet1d import ResNet1D
from test import model_test
from utils import csv_writer_acc_count_wpl, normalize_oper
from constants import DMD_group_number, TD_group_number, all_group_number, low_sample_rate, high_sample_rate, \
    TD_group_number_30, DMD_group_number_30, all_group_number_30, six_min_path_29, hundred_meter_path_26, L3_path_30, \
    exclude_list_100m, people30_dataset_path_list

from utils import *
from window import window_oper

NUM_WORKERS = 0


def one_fold_training(model, arg_list, number, patient_makers, window_labels, window_data, NORMALIZE, EPOCH, device,
                      GET_OUTPUT_PROB, save_dir, SAVE_IMAGE, LEARN_RATE, BATCH_SIZE):

    lock.acquire()

    if device is None:
        for i in range(NUM_OF_GPU):
            if used_gpu_list[i] < max_process_per_gpu:
                device = i
                break
    used_gpu_list[device] += 1
    lock.release()

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
    torch.cuda.empty_cache()
    # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    lock.acquire()
    used_gpu_list[device] -= 1
    lock.release()
    if SAVE_IMAGE:
        save_dir_epoch_acc = os.path.join(save_dir, 'epoch_acc')
        if not os.path.exists(os.path.join(save_dir_epoch_acc)):
            os.makedirs(os.path.join(save_dir_epoch_acc))
        np.save(os.path.join(save_dir_epoch_acc, number + '.npy'), output_values_lists)
        # save_img_epoch_acc(train_acc_epoch, test_acc_epoch, loss_epoch, EPOCH, save_dir, number)

    if GET_OUTPUT_PROB:
        save_dir_prob = os.path.join(save_dir, 'origin_output_prob')
        output_values_lists = np.array(output_values_lists)
        # print('output_values_lists shape: ', output_values_lists.shape)
        if not os.path.exists(save_dir_prob):
            os.makedirs(save_dir_prob)
        np.save(os.path.join(save_dir_prob, number + '.npy'), output_values_lists)
    return correct_percentage_test


def major_voting_in_datasets(model, arg_list, word_marker, lr, batch_size, epochs, window_step, window_size,
                             normalize, num_of_gpu, get_output_prob, save_image):
    datasets_path = 'dataset/30_dmd_data_set'
    # plot L1 -L5 to see if there is some one need excluded

    EPOCH = epochs
    WINDOW_STEP = window_step
    WINDOW_SIZE = window_size

    # get 7 original data cut in window here
    # get the model things here
    # need a structure for ['paitent number','Class number','L1', 'L2', 'L3', 'L4', 'L5', '100m', '6min'] like this has:DMD/TD/NONE value
    parameters_lists = []
    people_dataset_corrs_lists = []
    pool = Pool(NUM_OF_GPU * max_process_per_gpu)

    path_list = people30_dataset_path_list

    for i in range(len(path_list)):
        patient_makers, window_labels, window_data = window_oper(WINDOW_SIZE, WINDOW_STEP,
                                                                 dataset_marker='30',
                                                                 dataset_path=path_list[i], )
        for person in people_list:
            if not (person.paths[i] is None):
                save_dir = os.path.join('major_vote', word_marker)
                device = None
                parameters_lists.append([model, arg_list, person.N, patient_makers, window_labels, window_data,
                                         normalize, EPOCH, device, get_output_prob,
                                         save_dir, save_image, lr, batch_size])
                people_dataset_corrs = [person.N, i]
                people_dataset_corrs_lists.append(people_dataset_corrs)
    iter_para = iter(parameters_lists)

    pool_mapping_results = pool.starmap_async(one_fold_training, iter_para)
    pool.close()
    pool.join()
    i = 0
    for value in iter(pool_mapping_results.get()):
        person_N = people_dataset_corrs_lists[i][0]
        person_dataset = people_dataset_corrs_lists[i][1]
        person = [p for p in people_list if p.N == person_N][0]
        if value > 0.5:
            person.set_prediction(i=person_dataset, pred='C'+"{:.2f}".format(value))
        else:
            person.set_prediction(i=person_dataset, pred='W'+"{:.2f}".format(value))
        i += 1
    for person in people_list:
        person.prediction_to_csv(word_marker + '_vote.csv')

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
    # torch.multiprocessing.set_start_method('spawn')
    W_SIZE = 160
    W_STEP = 5
    import exp_variants
    from exp_variants import n_fold_exper

    total_gpu_num = 8
    used_gpu_list = torch.multiprocessing.Manager().list([0] * total_gpu_num)

    max_process_per_gpu = 1

    lock = torch.multiprocessing.Lock()
    NUM_OF_GPU = 7

    SAVE_IMAGE = True
    GET_OUTPUT_PROB = True
    EPOCHS = 100
    LEARN_RATE = 0.001
    BATCH_SIZE = 2048
    # exp_variants.total_gpu_num = 8
    # exp_variants.NUM_OF_GPU = 7
    # exp_variants.max_process_per_gpu = 1
    # exp_variants.used_gpu_list = torch.multiprocessing.Manager().list([0] * exp_variants.total_gpu_num)
    # exp_variants.lock = torch.multiprocessing.Lock()

    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6'
    # print('resnet_100_meter_walk norm true and 6 min true and false')

    # NUM_OF_GPU = exp_variants.NUM_OF_GPU
    model = ResNet1D
    # model = CNN_Pooling
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
    # N_OF_Module = 2
    # arg_list = [W_SIZE, N_OF_Module]
    NORMALIZE = True
    word_marker = 'major_vote_for_resnet98_gap4_incre4_norm_prob'
    major_voting_in_datasets(model, arg_list, word_marker, LEARN_RATE, BATCH_SIZE, EPOCHS, W_STEP, W_SIZE,
                             NORMALIZE, NUM_OF_GPU, get_output_prob=False, save_image=False)

    #
    # multiple_running(model, arg_list, LEARN_RATE, BATCH_SIZE, six_min_path_29,
    #                  NORMALIZE, SAVE_IMAGE, GET_OUTPUT_PROB, W_SIZE, W_STEP, EPOCHS, NUM_OF_GPU,
    #                  repeat_time=1, save_mul=False, save_dir_mul='./save_result_mul',
    #                  word_marker='CNN_POOL_NOF2_6_min' + str(W_SIZE) + '_step' + str(
    #                      W_STEP) + '_Norm_' + str(NORMALIZE))


