import pandas as pd

DMD_group_number = ['990012', '990015', '990016', '990023008', '990023010', '990023015']
TD_group_number = ['990014', '990017', '990018', '990023003', '990023011', '990023014']
all_group_number = ['990012', '990014', '990015', '990016', '990017', '990018', '990023003', '990023008',
                    '990023010', '990023011', '990023014', '990023015', ]
low_sample_rate = ['990012', '990014', '990015', '990016', '990017', '990018']
high_sample_rate = ['990023003', '990023008', '990023010', '990023011', '990023014', '990023015']

TD_group_number_30 = [
    '23046', '23003', '23035', '23014', '23031', '23018', '23040', '23033',
    '23034', '23038', '23036', '23011', '23022', '23039', '23013'
]
DMD_group_number_30 = [
    '23023', '23006', '23026', '230041', '23043', '23030', '23015', '23007',
    '23041', '23008', '23029', '23010', '23017', '23012', '23028'
]
all_group_number_30 = [
    '23023', '23006', '23026', '230041', '23043', '23030', '23015', '23007',
    '23041', '23008', '23029', '23010', '23017', '23012', '23028',
    '23046', '23003', '23035', '23014', '23031', '23018', '23040', '23033',
    '23034', '23038', '23036', '23011', '23022', '23039', '23013',

]

bad_sample_30 = ['23015', '23023', '23043', '23026', '23014', '23006', '23028', '23010']

# PATH

L3_path_12 = './dataset'

six_min_path_29 = r'./dataset/30_dmd_data_set/6-min-walk'
hundred_meter_path_26 = r'dataset/30_dmd_data_set/100-meter-walk'

L1_path_30 = r'./dataset/30_dmd_data_set/Speed-Calibration-L1'
L2_path_30 = r'./dataset/30_dmd_data_set/Speed-Calibration-L2'
L3_path_30 = r'./dataset/30_dmd_data_set/Speed-Calibration-L3'
L4_path_30 = r'./dataset/30_dmd_data_set/Speed-Calibration-L4'
L5_path_30 = r'./dataset/30_dmd_data_set/Speed-Calibration-L5'

people30_dataset_path_list = [L1_path_30, L2_path_30, L3_path_30, L4_path_30, L5_path_30, hundred_meter_path_26,
                              six_min_path_29]
dataset_name_list = ['L1', 'L2', 'L3', 'L4', 'L5', '100m', '6min']
# exclude_list
exclude_list_100m = ['23015', '23043']


def person_characteristic_save():
    ID = all_group_number
    label = [1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1]
    age = [5, 11, 12, 10, 3, 11, 12, 15, 7, 5, 9, 5]
    weight = [34.7, 38.4, 52.6, 38.5, 20, 44.5, 57.7, 63.7, 29.8, 20.3, 41.8, 22.9]
    height = [127, 147.6, 145, 124.5, 106.3, 144, 155.6, 153.3, 133, 119.3, 132.9, 111.8]
    NSAA_score = [31, 34, 29, 26, 31, 34, 34, 15, 13, 34, 34, 25]
    df = pd.DataFrame({'ID': ID, 'label': label, 'age': age, 'weight': weight, 'height': height, 'NSAA': NSAA_score})
    df.to_csv('dataset/person_characteristic.csv', index=False, sep=',')
