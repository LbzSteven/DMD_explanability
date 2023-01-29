import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from constants import DMD_group_number, TD_group_number, all_group_number, low_sample_rate, high_sample_rate, \
    TD_group_number_30, DMD_group_number_30, all_group_number_30


# for label DMD =1 TD= 0
# perform on downsample data
def window_oper(numbers=None, window_size=33, window_step=33, dataset='30'):
    paitent_makers = []
    labels = []
    datas = []
    if dataset == '12':
        path = "dataset/downsample"
        DMD = DMD_group_number
        numbers = all_group_number
    elif dataset == '30':
        path = "dataset/30_dmd_data_set/Speed-Calibration-L3"
        DMD = DMD_group_number_30
        numbers = all_group_number_30
    else:
        raise Exception("Sorry, dataset wrong")
    # if numbers is None:
    #     numbers = all_group_number
    for number in numbers:
        csv_data = pd.read_csv(os.path.join(path, number + '.csv'))
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
                datas.append(data)

    return [paitent_makers, labels, datas]


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
            np_data = np.array(csv_data)
            for i in range(3):
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


def person_characteristic_save():
    ID = all_group_number
    label = [1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1]
    age = [5, 11, 12, 10, 3, 11, 12, 15, 7, 5, 9, 5]
    weight = [34.7, 38.4, 52.6, 38.5, 20, 44.5, 57.7, 63.7, 29.8, 20.3, 41.8, 22.9]
    height = [127, 147.6, 145, 124.5, 106.3, 144, 155.6, 153.3, 133, 119.3, 132.9, 111.8]
    NSAA_score = [31, 34, 29, 26, 31, 34, 34, 15, 13, 34, 34, 25]
    df = pd.DataFrame({'ID': ID, 'label': label, 'age': age, 'weight': weight, 'height': height, 'NSAA': NSAA_score})
    df.to_csv('dataset/person_characteristic.csv', index=False, sep=',')


person_characteristic_save()
