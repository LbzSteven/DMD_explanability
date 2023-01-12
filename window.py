import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DMD_group_number = ['990012', '990015', '990016', '990023008', '990023010', '990023015']
TD_group_number = ['990014', '990017', '990018', '990023003', '990023011', '990023014']
all_group_number = ['990012', '990014', '990015', '990016', '990017', '990018', '990023008', '990023010', '990023015',
                    '990023003', '990023011', '990023014']
low_sample_rate = ['990012', '990014', '990015', '990016', '990017', '990018']
high_sample_rate = ['990023003', '990023008', '990023010', '990023011', '990023014', '990023015']


# for label DMD =1 TD= 0
# perform on downsample data
def window_oper(numbers=None, window_size=33, window_step=33):
    paitent_makers = []
    labels = []
    datas = []
    if numbers is None:
        numbers = all_group_number
    for number in numbers:
        csv_data = pd.read_csv(os.path.join("dataset/downsample", number + '.csv'))
        if number in DMD_group_number:
            label = 1
        else:
            label = 0
        for s in range(0, len(csv_data), window_step):

            window = csv_data.iloc[s:s + window_size, 1:]
            data = np.array(window)

            if data.shape[0] == window_size:

                paitent_makers.append(number)
                labels.append(label)
                datas.append(data)

    return [paitent_makers, labels, datas]
