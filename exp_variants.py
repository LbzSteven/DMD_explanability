import math
import os.path
import time

import pandas as pd
from torch.multiprocessing import Pool
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from DMDdataset import DMDDataset
from utils import *
from datetime import datetime
from test import model_test
from window import window_oper
from constants import *
from people_characteristic_builder import people_list
import torch
# from matplotlib import pyplot as plt
from torch import nn

NUM_WORKERS = 0
total_gpu_num = 8
max_process_per_gpu = 1
used_gpu_list = torch.multiprocessing.Manager().list([0] * total_gpu_num)
lock = torch.multiprocessing.Lock()
NUM_OF_GPU = 7
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

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_person_count = len(people_number)
    NUM_OF_POOL = math.ceil(len(people_number) / num_of_gpu)
    fold_i = 0
    for i_pool in range(NUM_OF_POOL):
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


def person_variance(model, arg_list, word_marker, lr, batch_size, epochs, window_step, window_size,
                    dataset_path, normalize, num_of_gpu, get_output_prob=False, save_image=False,
                    dataset_marker='30'):
    # init
    currentDateAndTime = datetime.now()
    currentTime = currentDateAndTime.strftime("%m_%d_%H_%M")
    save_dir = None

    EPOCH = epochs
    WINDOW_STEP = window_step
    WINDOW_SIZE = window_size

    patient_makers, window_labels, window_data = window_oper(WINDOW_SIZE, WINDOW_STEP,
                                                             dataset_marker=dataset_marker, dataset_path=dataset_path, )

    people_number = [i.split('.')[0] for i in os.listdir(dataset_path)]

    person_acc_lists = []
    # for number in people_number:
    #     value = one_fold_training(model, arg_list, number, patient_makers, window_labels, window_data,
    #                                      normalize, EPOCH, i, get_output_prob,
    #                                      save_dir, save_image, lr, batch_size)

    for number in people_number:
        pool = Pool(num_of_gpu)
        person_acc_list = []
        parameters_lists = []
        for i in range(num_of_gpu):
            parameters_lists.append([model, arg_list, number, patient_makers, window_labels, window_data,
                                     normalize, EPOCH, i, get_output_prob,
                                     save_dir, save_image, lr, batch_size])
        iter_para = iter(parameters_lists)
        pool_mapping_results = pool.starmap_async(one_fold_training, iter_para)
        pool.close()
        pool.join()
        for value in pool_mapping_results.get():
            person_acc_list.append(value)
        person_acc_lists.append(person_acc_list)

    print(person_acc_lists)
    person_acc_lists = np.array(person_acc_lists)
    mean_person = np.mean(person_acc_lists, axis=1)
    std_person = np.std(person_acc_lists, axis=1)
    min_person = np.min(person_acc_lists, axis=1)
    max_person = np.max(person_acc_lists, axis=1)
    for fold_i, number in enumerate(people_number):
        print('%s has mean of %.3f and std of %.3f, min:%.3f,max:%.3f'
              % (number, mean_person[fold_i], std_person[fold_i], min_person[fold_i], max_person[fold_i]))


def major_voting_in_datasets(model, arg_list, word_marker, lr, batch_size, epochs, window_step, window_size,
                             normalize, num_of_gpu, get_output_prob, save_image, TRAIN_JUST_ONE, ):
    datasets_path = 'dataset/30_dmd_data_set'
    # plot L1 -L5 to see if there is some one need excluded

    EPOCH = epochs
    WINDOW_STEP = window_step
    WINDOW_SIZE = window_size

    # get 7 original data cut in window here
    # get the model things here
    # need a structure for ['paitent number','Class number','L1', 'L2', 'L3', 'L4', 'L5', '100m', '6min'] like this has:DMD/TD/NONE value
    parameters_lists = []
    for i in range(len(people30_dataset_path_list)):
        patient_makers, window_labels, window_data = window_oper(WINDOW_SIZE, WINDOW_STEP,
                                                                 dataset_marker='30',
                                                                 dataset_path=people30_dataset_path_list[i], )
        for person in people_list:
            if not (person.paths[i] is None):
                save_dir = ''
                parameters_lists.append([model, arg_list, person.N, patient_makers, window_labels, window_data,
                                         normalize, EPOCH, device, get_output_prob,
                                         save_dir, save_image, lr, batch_size])


# major_voting_in_datasets()
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
