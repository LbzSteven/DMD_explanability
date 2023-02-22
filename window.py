import os
import numpy as np
import pandas as pd
from scipy.fftpack import fft
from constants import DMD_group_number, all_group_number, low_sample_rate, DMD_group_number_30, all_group_number_30, L3_path_30


def window_oper(window_size=33, window_step=33, dataset_marker='30', dataset_path=L3_path_30,
                ):
    paitent_makers = []
    labels = []
    window_data = []
    if dataset_marker == '12':

        DMD = DMD_group_number
        # people_number_list = all_group_number
    elif dataset_marker == '30':

        DMD = DMD_group_number_30
        # people_number_list = all_group_number_30
    else:
        raise Exception("Sorry, dataset picking wrong")

    if not os.path.exists(dataset_path):
        raise Exception("Sorry, dataset not creating yet")
    # if numbers is None:
    #     numbers = all_group_number

    people_number_list = [i.split('.')[0] for i in os.listdir(dataset_path)]
    for number in people_number_list:
        csv_data = pd.read_csv(os.path.join(dataset_path, number + '.csv'))
        np_data = np.array(csv_data)
        x = np.array(np_data[:, 1])
        y = np.array(np_data[:, 2])
        z = np.array(np_data[:, 3])
        # np_data[:, 1] = (x - min(x)) / (max(x) - min(x))
        # np_data[:, 2] = (y - min(y)) / (max(y) - min(y))
        # np_data[:, 3] = (z - min(z)) / (max(z) - min(z))
        if number in DMD:
            label = 1
        else:
            label = 0
        for s in range(0, len(csv_data), window_step):

            window = csv_data.iloc[s:s + window_size, 1:]
            # window = np_data[s:s + window_size, 1:]
            data = np.array(window)

            if data.shape[0] == window_size:
                paitent_makers.append(number)
                labels.append(label)
                data = np.moveaxis(data, 0, -1)
                window_data.append(data)
    labels = np.array(labels)
    window_data = np.array(window_data)
    return [paitent_makers, labels, window_data]


def window_oper_HS_3windows(numbers=None, window_size=33, window_step=33, dataset='30'):
    paitent_makers = []
    labels = []
    datas = []

    path = "dataset/truncate"
    DMD = DMD_group_number
    numbers = all_group_number
    for number in numbers:
        csv_data = pd.read_csv(os.path.join(path, number + '.csv'))

        if number in DMD:
            label = 1
        else:
            label = 0

        # low sample same as always
        if number in low_sample_rate:
            for s in range(0, len(csv_data), window_step):

                window = csv_data.iloc[s:s + window_size, 1:]
                # window = np_data[s:s + window_size, 1:]
                data = np.array(window)

                if data.shape[0] == window_size:
                    paitent_makers.append(number)
                    labels.append(label)
                    data = np.moveaxis(data, 0, -1)
                    datas.append(data)
        # high sample: downsample but down sample in three
        else:

            for i in range(3):
                np_data = np.array(csv_data)
                np_data = np_data[i::3, :]
                for s in range(0, len(csv_data), window_step):

                    # window = csv_data.iloc[s:s + window_size, 1:]
                    window = np_data[s:s + window_size, 1:]
                    data = np.array(window)

                    if data.shape[0] == window_size:
                        paitent_makers.append(number)
                        labels.append(label)
                        data = np.moveaxis(data, 0, -1)
                        datas.append(data)

    return [paitent_makers, labels, datas]


def window_FFT_oper(numbers=None, window_size=80, window_step=1, dataset='12'):
    paitent_makers = []
    labels = []
    window_data = []
    if dataset == '12':
        data_path = "dataset/downsample"
        DMD = DMD_group_number
        numbers = all_group_number
    elif dataset == '30':
        data_path = "dataset/30_dmd_data_set/Speed-Calibration-L3"
        DMD = DMD_group_number_30
        numbers = all_group_number_30
    else:
        raise Exception("Sorry, dataset picking wrong")

    if not os.path.exists(data_path):
        raise Exception("Sorry, dataset not creating yet")
    # if numbers is None:
    #     numbers = all_group_number
    for number in numbers:
        csv_data = pd.read_csv(os.path.join(data_path, number + '.csv'))
        np_data = np.array(csv_data)
        x = np.array(np_data[:, 1])
        y = np.array(np_data[:, 2])
        z = np.array(np_data[:, 3])

        # np_data[:, 1] = (x - min(x)) / (max(x) - min(x))
        # np_data[:, 2] = (y - min(y)) / (max(y) - min(y))
        # np_data[:, 3] = (z - min(z)) / (max(z) - min(z))
        if number in DMD:
            label = 1
        else:
            label = 0
        for s in range(0, len(csv_data), window_step):

            # window = csv_data.iloc[s:s + window_size, 1:]
            window = np_data[s:s + window_size, 1:]
            data = np.array(window)

            if data.shape[0] == window_size:
                paitent_makers.append(number)
                labels.append(label)

                x_FFT = fft(window[:, 0])[0:window_size // 2]
                y_FFT = fft(window[:, 1])[0:window_size // 2]
                z_FFT = fft(window[:, 2])[0:window_size // 2]
                data = np.array([x_FFT.real, x_FFT.imag, y_FFT.real, y_FFT.imag, z_FFT.real, z_FFT.imag])
                # data = np.moveaxis(data, 0, -1)
                window_data.append(data)

    return [paitent_makers, labels, window_data]

# person_characteristic_save()
