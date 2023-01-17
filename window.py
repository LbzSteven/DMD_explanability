import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DMD_group_number = ['990012', '990015', '990016', '990023008', '990023010', '990023015']
TD_group_number = ['990014', '990017', '990018', '990023003', '990023011', '990023014']
all_group_number = ['990012', '990014', '990015', '990016', '990017', '990018', '990023003', '990023008',
                    '990023010', '990023011', '990023014', '990023015', ]
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
