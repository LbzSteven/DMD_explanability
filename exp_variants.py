import math
import os.path
import time

from torch.multiprocessing import Pool

from main import one_fold_training
from utils import *
from datetime import datetime
from window import window_oper


# from matplotlib import pyplot as plt




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

# major_voting_in_datasets()
