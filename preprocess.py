import os

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.fft import fft, fftfreq
from constants import DMD_group_number, TD_group_number, all_group_number, low_sample_rate, high_sample_rate, \
    TD_group_number_30, DMD_group_number_30, all_group_number_30

dir_list = os.listdir("dataset")


# 100Hz
# X: vertical
# Y: mediolateral
# Z: anteroposterior
# 33Hz
# X: mediolateral
# Y: anteroposterior
# Z: vertical

def raw_to_vma(numbers=None):
    if numbers is None:
        numbers = all_group_number
    for number in numbers:
        csv_data = pd.read_csv(os.path.join("dataset", number + '.csv'))
        np_data = np.array(csv_data)
        ts = np.array(np_data[:, 0])

        if number in high_sample_rate:
            vertical = np.array(np_data[:, 1])
            mediolateral = np.array(np_data[:, 2])
            anteroposterior = np.array(np_data[:, 3])

        else:
            vertical = np.array(np_data[:, 3])
            mediolateral = np.array(np_data[:, 1])
            anteroposterior = np.array(np_data[:, 2])

        dataframe = pd.DataFrame({'ts': ts, 'v': vertical, 'm': mediolateral, 'a': anteroposterior})
        dataframe.to_csv(os.path.join("dataset", "vma", number + '.csv'), index=False, sep=',')
        if number in DMD_group_number:
            maker = 'DMD'
        else:
            maker = 'TD'


def vam_to_truncated(numbers=None):
    if numbers is None:
        numbers = all_group_number
    for number in numbers:
        csv_data = pd.read_csv(os.path.join("dataset/vma", number + '.csv'))
        np_data = np.array(csv_data)

        if number == '990012':
            ts = np.array(np_data[:-15, 0])
            vertical = np.array(np_data[:-15, 1])
            mediolateral = np.array(np_data[:-15, 2])
            anteroposterior = np.array(np_data[:-15, 3])

        elif number == '990017':
            ts = np.array(np_data[50:-33, 0])
            vertical = np.array(np_data[50:-33, 1])
            mediolateral = np.array(np_data[50:-33, 2])
            anteroposterior = np.array(np_data[50:-33, 3])

        elif number == '990018':
            ts = np.array(np_data[15:-50, 0])
            vertical = np.array(np_data[15:-50, 1])
            mediolateral = np.array(np_data[15:-50, 2])
            anteroposterior = np.array(np_data[15:-50, 3])

        elif number == '990023014':
            ts = np.array(np_data[150:, 0])
            vertical = np.array(np_data[150:, 1])
            mediolateral = np.array(np_data[150:, 2])
            anteroposterior = np.array(np_data[150:, 3])

        else:
            ts = np.array(np_data[:, 0])
            vertical = np.array(np_data[:, 1])
            mediolateral = np.array(np_data[:, 2])
            anteroposterior = np.array(np_data[:, 3])

        dataframe = pd.DataFrame({'ts': ts, 'v': vertical, 'm': mediolateral, 'a': anteroposterior})
        dataframe.to_csv(os.path.join("dataset", "truncate", number + '.csv'), index=False, sep=',')
        if number in DMD_group_number:
            maker = 'DMD'
        else:
            maker = 'TD'


def truncated_to_downsample():
    for number in high_sample_rate:
        csv_data = pd.read_csv(os.path.join("dataset/truncate", number + '.csv'))
        np_data = np.array(csv_data)
        # np_data = np_data[0::3, :]
        ts = np.array(np_data[0:-1:3, 0])
        vertical = np.array(np_data[0:-1:3, 1])
        mediolateral = np.array(np_data[0:-1:3, 2])
        anteroposterior = np.array(np_data[0:-1:3, 3])

        dataframe = pd.DataFrame({'ts': ts, 'v': vertical, 'm': mediolateral, 'a': anteroposterior})
        dataframe.to_csv(os.path.join("dataset", "downsample", number + '.csv'), index=False, sep=',')

    for number in low_sample_rate:
        csv_data = pd.read_csv(os.path.join("dataset/truncate", number + '.csv'))
        np_data = np.array(csv_data)
        ts = np.array(np_data[:, 0])
        vertical = np.array(np_data[:, 1])
        mediolateral = np.array(np_data[:, 2])
        anteroposterior = np.array(np_data[:, 3])
        dataframe = pd.DataFrame({'ts': ts, 'v': vertical, 'm': mediolateral, 'a': anteroposterior})
        dataframe.to_csv(os.path.join("dataset", "downsample", number + '.csv'), index=False, sep=',')


def Downsample_FFT_ZeroHighFreq_Inverse():
    for number in all_group_number:
        csv_data = pd.read_csv(os.path.join("dataset/truncate", number + '.csv'))
        np_data = np.array(csv_data)
        ts = np.array(np_data[:, 0])
        vertical = np.array(np_data[:, 1])
        mediolateral = np.array(np_data[:, 2])
        anteroposterior = np.array(np_data[:, 3])

        vertical = FFT_ZeroHighFreq_Inverse(vertical)
        mediolateral = FFT_ZeroHighFreq_Inverse(mediolateral)
        anteroposterior = FFT_ZeroHighFreq_Inverse(anteroposterior)

        dataframe = pd.DataFrame({'ts': ts, 'v': vertical, 'm': mediolateral, 'a': anteroposterior})
        dataframe.to_csv(os.path.join("dataset", "downsample", number + '.csv'), index=False, sep=',')


def FFT_ZeroHighFreq_Inverse(input_vector):
    # FFT

    # zero_out

    # inverse

    output_vector = 0
    return output_vector


# raw_to_vma()
# vam_to_truncated()
truncated_to_downsample()
