import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

dir_list = os.listdir("dataset")

# 100Hz
# X: vertical
# Y: mediolateral
# Z: anteroposterior
# 33Hz
# X: mediolateral
# Y: anteroposterior
# Z: vertical

DMD_group_number = ['990012', '990015', '990016', '990023008', '990023010', '990023015']
TD_group_number = ['990014', '990017', '990018', '990023003', '990023011', '990023014']
all_group_number = ['990012', '990014', '990015', '990016', '990017', '990018', '990023008', '990023010', '990023015',
                    '990023003', '990023011', '990023014']
low_sample_rate = ['990012', '990014', '990015', '990016', '990017', '990018']
high_sample_rate = ['990023003', '990023008', '990023010', '990023011', '990023014', '990023015']


def get_data_label(numbers=None):
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


get_data_label()
