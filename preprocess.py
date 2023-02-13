import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq, ifft
from constants import DMD_group_number, TD_group_number, all_group_number, low_sample_rate, high_sample_rate, \
    TD_group_number_30, DMD_group_number_30, all_group_number_30, bad_sample_30, six_min_path_30, hundred_meter_path_26

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


def FFT_freq_plot(magnitude, magnitude_zero_out, input_vector, inverse_vector, output_vector, freq_half,
                  label='990012'):
    plt.figure(figsize=(24, 15))
    ax = plt.subplot(511)
    ax.plot(freq_half, magnitude, label='FFT magnitude')
    ax.set_xlabel('freq')
    ax.set_ylabel('magnitude')
    ax.set_title('FFT freq ' + label)
    ax.grid()
    ax.legend()

    ax = plt.subplot(512)
    ax.plot(freq_half, magnitude_zero_out, label='FFT magnitude zero out')
    ax.set_xlabel('freq')
    ax.set_ylabel('magnitude')
    ax.set_title('FFT freq zero out of' + label)
    ax.grid()
    ax.legend()
    #
    ax = plt.subplot(513)
    ax.plot(range(input_vector.shape[0]), input_vector, label='original signal')
    ax.set_xlabel('time')
    ax.set_ylabel('accelerator')
    ax.set_title('original signal ' + label)
    ax.grid()
    ax.legend()
    #
    ax = plt.subplot(514)
    ax.plot(range(inverse_vector.shape[0]), inverse_vector, label='inverse signal')
    ax.set_xlabel('time')
    ax.set_ylabel('accelerator')
    ax.set_title('inversed signal ' + label)
    ax.grid()
    ax.legend()
    #
    ax = plt.subplot(515)
    ax.plot(range(output_vector.shape[0]), output_vector, label='zero out inverse signal')
    ax.set_xlabel('accelerator')
    ax.set_ylabel('time')
    ax.set_title('inversed zero-out signal ' + label)
    ax.grid()
    ax.legend()
    plt.show()


def Downsample_FFT_ZeroHighFreq_Inverse(save_path, dataset_path=hundred_meter_path_26, time_interval=0.01, zero_out_freq_limit=30
                                        ,
                                        sanity_check=False):
    if save_path is None:
        save_path = os.path.join('dataset/ZeroHighFreq/', 'hundred_meter_26people_freq_' + str(zero_out_freq_limit))
    time_interval = time_interval
    people_number = [i.split('.')[0] for i in os.listdir(dataset_path)]
    if not os.path.exists(dataset_path):
        raise Exception('wrong dataset input')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if sanity_check:
        people_number = ['23015']

    for number in people_number:
        csv_data = pd.read_csv(os.path.join(dataset_path, number + '.csv'))
        np_data = np.array(csv_data)
        ts = np.array(np_data[:, 0])
        vertical = np.array(np_data[:, 1])
        mediolateral = np.array(np_data[:, 2])
        anteroposterior = np.array(np_data[:, 3])
        if sanity_check:
            FFT_ZeroHighFreq_Inverse(vertical, time_interval, plotting=True)
        else:
            vertical = FFT_ZeroHighFreq_Inverse(vertical, time_interval, plotting=False)
            mediolateral = FFT_ZeroHighFreq_Inverse(mediolateral, time_interval, plotting=False)
            anteroposterior = FFT_ZeroHighFreq_Inverse(anteroposterior, time_interval, plotting=False)

            dataframe = pd.DataFrame({'ts': ts, 'v': vertical, 'm': mediolateral, 'a': anteroposterior})
            dataframe.to_csv(os.path.join(save_path, number + '.csv'), index=False, sep=',')


def FFT_ZeroHighFreq_Inverse(input_vector, time_interval=0.01, zero_out_freq_limit=6, plotting=False):
    # FFT
    time_interval = time_interval  # 0.01
    sample_frequent = 1 / time_interval  # 33
    fft_result = fft(input_vector)
    magnitude = np.abs(fft_result[0:input_vector.shape[0] // 2])
    freq = fftfreq(input_vector.shape[0], time_interval)
    freq_half = freq[: input_vector.shape[0] // 2]
    # zero_out
    # magnitude_zero_out = magnitude
    # magnitude_zero_out[freq_half > 7.5] = 0

    inverse_vector = ifft(fft_result)
    output_vector = np.where(((freq < zero_out_freq_limit) & (-zero_out_freq_limit < freq)), fft_result, 0)

    # inverse
    magnitude_zero_out = np.abs(output_vector[0:input_vector.shape[0] // 2])
    output_vector = ifft(output_vector).real

    # sanity check
    if plotting:
        FFT_freq_plot(magnitude, magnitude_zero_out, input_vector, inverse_vector, output_vector, freq_half, 'X')
    return output_vector


# raw_to_vma()
# vam_to_truncated()
# truncated_to_downsample()
Downsample_FFT_ZeroHighFreq_Inverse(None)
