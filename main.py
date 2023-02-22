import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import exp_variants
from exp_variants import multiple_running
from CNN_torch import CNN_DMD, CNN_var, CNN_Pooling, one_axis_CNN, CNN_for_window_FFT
from utils import csv_writer_acc_count_wpl
from constants import DMD_group_number, TD_group_number, all_group_number, low_sample_rate, high_sample_rate, \
    TD_group_number_30, DMD_group_number_30, all_group_number_30, six_min_path_29, hundred_meter_path_26, L3_path_30, \
    exclude_list_100m

from utils import *



if __name__ == "__main__":
    # multi running to get limit
    W_SIZE = 160
    W_STEP = 5

    SAVE_IMAGE = True
    GET_OUTPUT_PROB = True
    EPOCHS = 100
    LEARN_RATE = 0.001
    BATCH_SIZE = 2048
    exp_variants.total_gpu_num = 8
    exp_variants.NUM_OF_GPU = 7
    exp_variants.max_process_per_gpu = 1
    exp_variants.used_gpu_list = torch.multiprocessing.Manager().list([0] * exp_variants.total_gpu_num)
    exp_variants.lock = torch.multiprocessing.Lock()

    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6'
    # print('resnet_100_meter_walk norm true and 6 min true and false')
    torch.multiprocessing.set_start_method('spawn')

    model = CNN_Pooling
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

    # arg_list = [in_channels, base_filters, kernel_size, stride, groups, n_block, n_classes, downsample_gap,
    #             increasefilter_gap, use_bn, use_do, verbose]\
    N_OF_Module = 2
    arg_list = [W_SIZE, N_OF_Module]
    NORMALIZE = True

    multiple_running(model, arg_list, LEARN_RATE, BATCH_SIZE, six_min_path_29,
                     NORMALIZE, SAVE_IMAGE, GET_OUTPUT_PROB, W_SIZE, W_STEP, EPOCHS, NUM_OF_GPU,
                     repeat_time=1, save_mul=False, save_dir_mul='./save_result_mul',
                     word_marker='CNN_POOL_NOF2_6_min' + str(W_SIZE) + '_step' + str(
                         W_STEP) + '_Norm_' + str(NORMALIZE))
    multiple_running(model, arg_list, LEARN_RATE, BATCH_SIZE, hundred_meter_path_26,
                     NORMALIZE, SAVE_IMAGE, GET_OUTPUT_PROB, W_SIZE, W_STEP, EPOCHS, NUM_OF_GPU,
                     repeat_time=1, save_mul=False, save_dir_mul='./save_result_mul',
                     word_marker='CNN_POOL_NOF2_100_meter' + str(W_SIZE) + '_step' + str(
                         W_STEP) + '_Norm_' + str(NORMALIZE))
    NORMALIZE = False
    multiple_running(model, arg_list, LEARN_RATE, BATCH_SIZE, six_min_path_29,
                     NORMALIZE, SAVE_IMAGE, GET_OUTPUT_PROB, W_SIZE, W_STEP, EPOCHS, NUM_OF_GPU,
                     repeat_time=1, save_mul=False, save_dir_mul='./save_result_mul',
                     word_marker='CNN_POOL_NOF2_6_min' + str(W_SIZE) + '_step' + str(
                         W_STEP) + '_Norm_' + str(NORMALIZE))
    multiple_running(model, arg_list, LEARN_RATE, BATCH_SIZE, hundred_meter_path_26,
                     NORMALIZE, SAVE_IMAGE, GET_OUTPUT_PROB, W_SIZE, W_STEP, EPOCHS, NUM_OF_GPU,
                     repeat_time=1, save_mul=False, save_dir_mul='./save_result_mul',
                     word_marker='CNN_POOL_NOF2_100_meter' + str(W_SIZE) + '_step' + str(
                         W_STEP) + '_Norm_' + str(NORMALIZE))

    N_OF_Module = 3
    arg_list = [W_SIZE, N_OF_Module]

    NORMALIZE = True
    multiple_running(model, arg_list, LEARN_RATE, BATCH_SIZE, six_min_path_29,
                     NORMALIZE, SAVE_IMAGE, GET_OUTPUT_PROB, W_SIZE, W_STEP, EPOCHS, NUM_OF_GPU,
                     repeat_time=1, save_mul=False, save_dir_mul='./save_result_mul',
                     word_marker='CNN_POOL_NOF3_6_min' + str(W_SIZE) + '_step' + str(
                         W_STEP) + '_Norm_' + str(NORMALIZE))
    multiple_running(model, arg_list, LEARN_RATE, BATCH_SIZE, hundred_meter_path_26,
                     NORMALIZE, SAVE_IMAGE, GET_OUTPUT_PROB, W_SIZE, W_STEP, EPOCHS, NUM_OF_GPU,
                     repeat_time=1, save_mul=False, save_dir_mul='./save_result_mul',
                     word_marker='CNN_POOL_NOF3_100_meter' + str(W_SIZE) + '_step' + str(
                         W_STEP) + '_Norm_' + str(NORMALIZE))
    NORMALIZE = False
    multiple_running(model, arg_list, LEARN_RATE, BATCH_SIZE, six_min_path_29,
                     NORMALIZE, SAVE_IMAGE, GET_OUTPUT_PROB, W_SIZE, W_STEP, EPOCHS, NUM_OF_GPU,
                     repeat_time=1, save_mul=False, save_dir_mul='./save_result_mul',
                     word_marker='CNN_POOL_NOF3_6_min' + str(W_SIZE) + '_step' + str(
                         W_STEP) + '_Norm_' + str(NORMALIZE))
    multiple_running(model, arg_list, LEARN_RATE, BATCH_SIZE, hundred_meter_path_26,
                     NORMALIZE, SAVE_IMAGE, GET_OUTPUT_PROB, W_SIZE, W_STEP, EPOCHS, NUM_OF_GPU,
                     repeat_time=1, save_mul=False, save_dir_mul='./save_result_mul',
                     word_marker='CNN_POOL_NOF3_100_meter' + str(W_SIZE) + '_step' + str(
                         W_STEP) + '_Norm_' + str(NORMALIZE))

    N_OF_Module = 4
    arg_list = [W_SIZE, N_OF_Module]
    NORMALIZE = True
    multiple_running(model, arg_list, LEARN_RATE, BATCH_SIZE, six_min_path_29,
                     NORMALIZE, SAVE_IMAGE, GET_OUTPUT_PROB, W_SIZE, W_STEP, EPOCHS, NUM_OF_GPU,
                     repeat_time=1, save_mul=False, save_dir_mul='./save_result_mul',
                     word_marker='CNN_POOL_NOF4_6_min' + str(W_SIZE) + '_step' + str(
                         W_STEP) + '_Norm_' + str(NORMALIZE))
    multiple_running(model, arg_list, LEARN_RATE, BATCH_SIZE, hundred_meter_path_26,
                     NORMALIZE, SAVE_IMAGE, GET_OUTPUT_PROB, W_SIZE, W_STEP, EPOCHS, NUM_OF_GPU,
                     repeat_time=1, save_mul=False, save_dir_mul='./save_result_mul',
                     word_marker='CNN_POOL_NOF4_100_meter' + str(W_SIZE) + '_step' + str(
                         W_STEP) + '_Norm_' + str(NORMALIZE))
    NORMALIZE = False
    multiple_running(model, arg_list, LEARN_RATE, BATCH_SIZE, six_min_path_29,
                     NORMALIZE, SAVE_IMAGE, GET_OUTPUT_PROB, W_SIZE, W_STEP, EPOCHS, NUM_OF_GPU,
                     repeat_time=1, save_mul=False, save_dir_mul='./save_result_mul',
                     word_marker='CNN_POOL_NOF4_6_min' + str(W_SIZE) + '_step' + str(
                         W_STEP) + '_Norm_' + str(NORMALIZE))
    multiple_running(model, arg_list, LEARN_RATE, BATCH_SIZE, hundred_meter_path_26,
                     NORMALIZE, SAVE_IMAGE, GET_OUTPUT_PROB, W_SIZE, W_STEP, EPOCHS, NUM_OF_GPU,
                     repeat_time=1, save_mul=False, save_dir_mul='./save_result_mul',
                     word_marker='CNN_POOL_NOF4_100_meter' + str(W_SIZE) + '_step' + str(
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
