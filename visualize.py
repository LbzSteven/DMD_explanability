import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy.fft import fft, fftfreq
from statsmodels.tsa.stattools import acf
import seaborn as sns

dir_list = os.listdir("dataset")
# print(len(dir_list))
# print(dir_list)


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


def the_length_of_data(labels=None):
    if labels is None:
        labels = all_group_number
    for label in labels:
        csv_data = pd.read_csv(os.path.join("dataset", label + '.csv'))
        np_data = np.array(csv_data)
        ts = np.array(np_data[:, 0])
        if label in DMD_group_number:
            maker = 'DMD'
        else:
            maker = 'TD'
        print("number '" + label + "'(" + maker + ") has " + str(ts.shape[0]) + ' sample points')


def raw_data_image():
    for csv_file_name in dir_list:
        csv_data = pd.read_csv(os.path.join("dataset", csv_file_name))

        np_data = np.array(csv_data)
        # np_data = np_data[:100, :]
        x = np.array(np_data[:, 1])
        y = np.array(np_data[:, 2])
        z = np.array(np_data[:, 3])
        fig, ax = plt.subplots()

        plot_x_axis = range(np_data.shape[0])

        if csv_file_name.split('.')[0] in high_sample_rate:
            vertical = x
            mediolateral = y
            anteroposterior = z
        else:
            vertical = z
            mediolateral = x
            anteroposterior = y
        print('The mean of ' + csv_file_name.split('.')[0])
        print(np.mean(vertical))
        print(np.mean(mediolateral))
        print(np.mean(anteroposterior))
        ax.plot(plot_x_axis, vertical, label='vertical')
        ax.plot(plot_x_axis, mediolateral, label='mediolateral')
        ax.plot(plot_x_axis, anteroposterior, label='anteroposterior')
        ax.legend()
        ax.set_xlabel("record")
        ax.set_ylabel("sensor data")

        plt.savefig(os.path.join('./visualize/raw_data', csv_file_name.split('.')[0]))


def normalized_data_image_all(labels=None):  # wrong way to do it
    if labels is None:
        labels = all_group_number
    x_list = []
    y_list = []
    z_list = []
    ts_list = []
    for label in labels:
        csv_data = pd.read_csv(os.path.join("dataset", label + '.csv'))
        np_data = np.array(csv_data)

        ts = np.array(np_data[:, 0])
        x = np.array(np_data[:, 1])
        y = np.array(np_data[:, 2])
        z = np.array(np_data[:, 3])

        ts_list.append(ts)
        x_list.append(x)
        y_list.append(y)
        z_list.append(z)

    max_x = max(max(row) for row in x_list)
    min_x = min(min(row) for row in x_list)

    max_y = max(max(row) for row in y_list)
    min_y = min(min(row) for row in y_list)

    max_z = max(max(row) for row in z_list)
    min_z = min(min(row) for row in z_list)

    for i in range(len(labels)):
        ts = ts_list[i]
        x = np.array(x_list[i])
        x = (x - max_x) / (max_x - min_x)
        y = np.array(y_list[i])
        y = (y - max_y) / (max_y - min_y)
        z = np.array(z_list[i])
        z = (z - max_z) / (max_z - min_z)
        fig, ax = plt.subplots(figsize=(10, 6), dpi=400)
        ax.plot(ts, x, label='x')
        ax.plot(ts, y, label='y')
        ax.plot(ts, z, label='z')
        ax.legend()
        ax.set_xlabel("time series")
        ax.set_ylabel("sensor data")
        if label in DMD_group_number:
            plt.savefig(os.path.join('./visualize/normalized_data', label) + ' DMD')
        else:
            plt.savefig(os.path.join('./visualize/normalized_data', label) + ' Control')


def time_intervals_checking():
    for label in all_group_number:
        csv_data = pd.read_csv(os.path.join("dataset", label + '.csv'))
        # np_data = np.array(csv_data)
        np_data = np.array(csv_data)
        ts = np.array(np_data[:, 0])
        average_intervals = (ts[-1] - ts[0]) / (ts.shape[0] - 1)
        sample_rate = (ts.shape[0] - 1) / (ts[-1] - ts[0])
        variance = 0
        for i in range(ts.shape[0] - 1):
            variance += (ts[i + 1] - ts[i] - average_intervals) ** 2
        variance = variance / (ts.shape[0] - 1)
        print("average intervals for:", label, "is", average_intervals)
        print("variance for intervals:", label, "is", variance)
        print("sample rate:", label, "is", sample_rate)


def mapping_same_x_for_first_six():
    x_axis_dict = []
    name_dict = []
    length_dict = []
    for csv_file_name in dir_list:
        if csv_file_name.split('.')[0] in ['990012', '990015', '990016', '990017', '990018', '990014']:
            csv_data = pd.read_csv(os.path.join("dataset", csv_file_name))
            np_data = np.array(csv_data)
            x = np.array(np_data[:, 1])
            # x = (x - min(x)) / (max(x) - min(x))
            x_axis_dict.append(x)
            name_dict.append(csv_file_name.split('.')[0])
            length_dict.append(x.shape)
    # print(x_axis_dict)
    # print(name_dict)
    fig, ax = plt.subplots(figsize=(15, 6))
    min_length = min(length_dict)
    print(name_dict)
    print(length_dict)
    print(min_length)
    i = 0
    for x_axis in x_axis_dict:
        x_axis = x_axis[0:(min_length[0])]
        ax.plot(range(x_axis.shape[0]), x_axis, label=name_dict[i])
        print(name_dict[i])
        i += 1
    ax.legend()
    # plt.show()
    plt.savefig("1.jpg")


def visualize_raw_x_axis(sample_number=100, label_set=None):
    if label_set is None:
        label_set = ['990012', '990015', '990016', '990017', '990018', '990014']
    x_axis_dict = []
    name_dict = []
    length_dict = []
    for csv_file_name in dir_list:
        if csv_file_name.split('.')[0] in label_set:
            csv_data = pd.read_csv(os.path.join("dataset", csv_file_name))
            np_data = np.array(csv_data)
            x = np.array(np_data[0:sample_number, 1])
            x_axis_dict.append(x)
            name_dict.append(csv_file_name.split('.')[0])
            length_dict.append(x.shape)
    fig, ax = plt.subplots(figsize=(15, 6))
    min_length = min(length_dict)
    print(name_dict)
    print(length_dict)
    print(min_length)
    i = 0
    for x_axis in x_axis_dict:
        x_axis = x_axis[0:(min_length[0])]
        ax.plot(range(x_axis.shape[0]), x_axis, label=name_dict[i])
        print(name_dict[i])
        i += 1
    ax.legend()
    plt.savefig("visualize_x_axis_12_14_" + str(sample_number))


def visualize_one_person(sample_number=None, labels=None):
    if labels is None:
        labels = ['990012']
    if sample_number is None:
        sample_number = 100
    for label in labels:
        csv_data = pd.read_csv(os.path.join("dataset", label + '.csv'))
        np_data = np.array(csv_data)
        ts = np.array(np_data[0:sample_number, 0])
        x = np.array(np_data[0:sample_number, 1])
        # x = (x - min(x)) / (max(x) - min(x))
        y = np.array(np_data[0:sample_number, 2])
        # y = (y - min(y)) / (max(y) - min(y))
        z = np.array(np_data[0:sample_number, 3])
        # z = (z - min(z)) / (max(z) - min(z))
        fig, ax = plt.subplots(figsize=(10, 6), dpi=400)
        if label in high_sample_rate:
            vertical = x
            mediolateral = y
            anteroposterior = z
        else:
            vertical = z
            mediolateral = x
            anteroposterior = y

        ax.plot(ts, vertical, label='vertical')
        ax.plot(ts, mediolateral, label='mediolateral')
        ax.plot(ts, anteroposterior, label='anteroposterior')
        ax.legend()
        ax.set_xlabel("time series")
        ax.set_ylabel("sensor data")

        if label in DMD_group_number:
            plt.savefig(os.path.join('./visualize/raw_data', label + '_' + str(sample_number)) + ' DMD')
        else:
            plt.savefig(os.path.join('./visualize/raw_data', label + '_' + str(sample_number)) + ' Control')


def FFT(data, sample_number, sample_spacing, label='990012', axis_marker='x', top_number_K_frequency=5):
    # if False:
    if label in high_sample_rate:
        file_dir = os.path.join('./visualize/FFT/downsample', label)

    else:
        file_dir = os.path.join('./visualize/FFT', label)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    txt_dir = os.path.join(file_dir, label + '_' + axis_marker + '_FFT_result.txt')
    # if False:
    if os.path.exists(txt_dir):
        yf = np.loadtxt(txt_dir, delimiter=',').view(complex).reshape(-1)
    else:
        yf = fft(data)
        np.savetxt(txt_dir, yf.view(float).reshape(-1, 2), delimiter=',')

    spacing = sample_spacing
    number = sample_number

    xf = fftfreq(number, spacing)[:number // 2]
    angles = np.angle(yf[:number // 2], deg=True)
    yf = np.abs(yf[0:number // 2])
    yf = yf / number

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(xf, yf)
    ax.set_xlabel('fft freq')
    ax.set_ylabel('fft magnitude')
    ax.grid()

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    plt.savefig(os.path.join(file_dir, label + '_' + axis_marker + ''))

    # plt.show()
    # print(xf)
    top_k = top_number_K_frequency
    # top K=3 index
    top_k_idxs = np.argpartition(yf, -top_k)[-top_k:]
    top_k_yf = yf[top_k_idxs]
    # fft_periods = (1 / xf[top_k_idxs])
    # print(f"top_yf of {marker}: {top_k_yf}")
    print(f"fft rate of {axis_marker}: {xf[top_k_idxs]}")
    print(f'fft phase of {axis_marker}: {angles[top_k_idxs]}')

    # ax.plot(top_k_idxs, yf[top_k_idxs] ,label='score')
    fig, ax2 = plt.subplots(figsize=(15, 6))
    ax2.plot(xf, angles, label='angle')
    ax2.set_xlabel('fft freq')
    ax2.set_ylabel('fft phase')
    ax2.grid()
    ax2.legend()
    plt.savefig(os.path.join(file_dir, label + '_phase_' + axis_marker))

    fig, ax2 = plt.subplots(figsize=(15, 6))
    ax2.plot(xf, yf, label='magnitude' + axis_marker)
    ax2.set_xlabel('fft freq')
    ax2.set_ylabel('fft magnitude on ' + axis_marker)
    ax2.grid()
    ax2.legend()
    plt.savefig(os.path.join(file_dir, label + '_' + axis_marker))

    # power = np.abs(fft_series)
    # sample_freq = fftfreq(fft_series.size)
    # pos_mask = np.where(sample_freq > 0)
    # freqs = sample_freq[pos_mask]
    # powers = power[pos_mask]

    # ax.plot(range(fft_series.shape[0]), fft_series)
    # plt.show()

    # xf = fftfreq(number)[:number // 2]
    # ax.plot(range(xf.shape[0]), xf)
    # plt.show()

    # acf_scores = []
    # for lag in top_k_idxs:
    #     # lag = fft_periods[np.abs(fft_periods - time_lag).argmin()]
    #     acf_score = acf(data, nlags=lag)[-1]
    #     acf_scores.append(acf_score)
    #     # print(f"lag: {lag} fft acf: {acf_score}")
    # print(f"lag: {top_k_idxs[acf_scores.index(max(acf_scores))]} has highest fft acf: {max(acf_scores)}")


def frequency_by_FFT(labels=None):
    if labels is None:
        labels = ['990012']
    for label in labels:
        csv_data = pd.read_csv(os.path.join("dataset", label + '.csv'))
        np_data = np.array(csv_data)
        ts = np.array(np_data[:, 0])
        x = np.array(np_data[:, 1])
        y = np.array(np_data[:, 2])
        z = np.array(np_data[:, 3])
        # sanity check
        # a = np.arange(1, 11, 1)
        # print(a.size)
        # print(a[0:a.size:3])
        # if False:
        if label in high_sample_rate:
            ts = ts[0:ts.size:3]
            x = x[0:x.size:3]
            y = y[0:y.size:3]
            z = z[0:z.size:3]
            vertical = x
            mediolateral = y
            anteroposterior = z
        else:
            vertical = z
            mediolateral = x
            anteroposterior = y
        print("The frequency an phase of '" + label + "'")
        sample_number = ts.shape[0]
        sample_spacing = (ts[-1] - ts[0]) / ts.shape[0]
        # sample_spacing = 0.033

        # test if I am right
        # samples = np.linspace(0, 1, 1400)
        # y = 1 * np.sin(2 * np.pi * 200 * samples) +\
        #     2 * np.sin(2 * np.pi * 400 * samples) + \
        #     3 * np.sin(2 * np.pi * 600 * samples)
        # FFT(y, 1400, 1 / 600, 'test', 'test')
        # FFT(vertical, sample_number, sample_spacing, label, 'vertical')
        # FFT(mediolateral, sample_number, sample_spacing, label, 'mediolateral')
        # FFT(anteroposterior, sample_number, sample_spacing, label, 'anteroposterior')
        # if False:
        if label in high_sample_rate:
            file_dir = os.path.join('./visualize/FFT/downsample', label)
        else:
            file_dir = os.path.join('./visualize/FFT/', label)
        FFT_vertical = np.loadtxt(os.path.join(file_dir, label + '_vertical_FFT_result.txt'), delimiter=',').view(
            complex).reshape(
            -1)[:ts.shape[0] // 2]

        FFT_mediolateral = np.loadtxt(os.path.join(file_dir, label + '_mediolateral_FFT_result.txt'),
                                      delimiter=',').view(complex).reshape(
            -1)[:ts.shape[0] // 2]

        FFT_anteroposterior = np.loadtxt(os.path.join(file_dir, label + '_anteroposterior_FFT_result.txt'),
                                         delimiter=',').view(complex).reshape(
            -1)[:ts.shape[0] // 2]

        fig, ax = plt.subplots(figsize=[15, 6])
        xf = fftfreq(ts.shape[0], (ts[-1] - ts[0]) / ts.shape[0])[:ts.shape[0] // 2]
        ax.plot(xf, np.abs(FFT_vertical) / ts.shape[0], label='magnitude on vertical')
        ax.plot(xf, np.abs(FFT_mediolateral) / ts.shape[0], label='magnitude on mediolateral')
        ax.plot(xf, np.abs(FFT_anteroposterior) / ts.shape[0], label='magnitude on anteroposterior')
        ax.set_xlabel('Frequency(Hz)')
        ax.set_ylabel('magnitude of each frequency on all axis')
        ax.set_title('Compare all axis in ' + label)
        ax.legend()
        # plt.savefig(os.path.join(file_dir,'all_axis_magnitude'))
        plt.savefig(os.path.join('visualize/FFT/all_axis_magnitude', label))


def compare_frequency(labels=None):
    if labels is None:
        labels = ['990012']

    axis_markers = ['vertical', 'mediolateral', 'anteroposterior']
    yf_list_vertical = []
    yf_list_mediolateral = []
    yf_list_anteroposterior = []
    xf_list = []
    for label in labels:
        if label in high_sample_rate:
            file_dir = os.path.join('./visualize/FFT/downsample', label)
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
        else:
            file_dir = os.path.join('./visualize/FFT', label)

        csv_data = pd.read_csv(os.path.join("dataset", label + '.csv'))
        np_data = np.array(csv_data)
        ts = np.array(np_data[:, 0])
        if label in high_sample_rate:
            ts = ts[0:ts.size:3]
        for axis_marker in axis_markers:
            txt_dir = os.path.join(file_dir, label + '_' + axis_marker + '_FFT_result.txt')
            # if False:
            if os.path.exists(txt_dir):
                yf = np.loadtxt(txt_dir, delimiter=',').view(complex).reshape(-1)[:ts.shape[0] // 2]
                yf = yf / ts.shape[0]

                if axis_marker == 'vertical':
                    yf_list_vertical.append(yf)
                elif axis_marker == 'mediolateral':
                    yf_list_mediolateral.append(yf)
                else:
                    yf_list_anteroposterior.append(yf)

            else:
                if axis_marker == 'vertical':
                    data = np.array(np_data[:, 1])
                    yf = fft(data)[:ts.shape[0] // 2]
                    yf = yf / ts.shape[0]
                    np.savetxt(txt_dir, yf.view(float).reshape(-1, 2), delimiter=',')
                    yf_list_vertical.append(yf)
                elif axis_marker == 'mediolateral':
                    data = np.array(np_data[:, 2])
                    yf = fft(data)[:ts.shape[0] // 2]
                    yf = yf / ts.shape[0]
                    np.savetxt(txt_dir, yf.view(float).reshape(-1, 2), delimiter=',')
                    yf_list_mediolateral.append(yf)
                else:
                    data = np.array(np_data[:, 3])
                    yf = fft(data)[:ts.shape[0] // 2]
                    yf = yf / ts.shape[0]
                    np.savetxt(txt_dir, yf.view(float).reshape(-1, 2), delimiter=',')
                    yf_list_anteroposterior.append(yf)

        xf = fftfreq(ts.shape[0], (ts[-1] - ts[0]) / ts.shape[0])[:ts.shape[0] // 2]
        xf_list.append(xf)
    # DMD_color = ['deepskyblue', 'skyblue', 'lightskyblue']
    # TD_color = ['darkorange', 'orange', 'gold']
    DMD_color = ['deepskyblue', 'darkcyan', 'darkgreen']
    TD_color = ['darkorange', 'brown', 'violet']

    plt.figure(figsize=(18, 15))
    plt.ylabel('Magnitude on ', fontsize=20)
    j = 0
    for axis_marker in axis_markers:
        ax = plt.subplot(311 + j)
        j += 1
        number_of_DMD = 0
        for i in range(len(xf_list)):

            if labels[i] in DMD_group_number:
                marker = 'DMD'
                color = DMD_color[number_of_DMD]
                number_of_DMD += 1
            else:
                marker = 'TD'
                color = TD_color[i - number_of_DMD]
            if axis_marker == 'vertical':
                ax.plot(xf_list[i], np.abs(yf_list_vertical[i]), c=color, label=marker + ' ' + labels[i])
            elif axis_marker == 'mediolateral':
                ax.plot(xf_list[i], np.abs(yf_list_mediolateral[i]), c=color, label=marker + ' ' + labels[i])
            else:
                ax.plot(xf_list[i], np.abs(yf_list_anteroposterior[i]), c=color, label=marker + ' ' + labels[i])

            ax.set_ylabel(axis_marker, fontsize=20)
            ax.set_title(axis_marker, fontsize=20)
            ax.legend()
    plt.suptitle('Compare all axis in ' + str(labels), fontsize=20)
    plt.xlabel('Frequency(Hz)', fontsize=20)

    # plt.show()
    plt.savefig(os.path.join('./visualize/frequency compare', str(labels)))


def pearson(labels=None):
    if labels is None:
        labels = ['990012']
    for label in labels:
        csv_data = pd.read_csv(os.path.join("dataset", label + '.csv'))
        overall_pearson_r = csv_data.corr()
        print(overall_pearson_r)
        plt.figure(figsize=(12, 8))
        sns.heatmap(overall_pearson_r, cmap="Greens", annot=True)


def visualize_vma(sample_number=100, position='truncate'):
    for label in all_group_number:


        csv_data = pd.read_csv(os.path.join("dataset", label + '.csv'))
        np_data = np.array(csv_data)

        if position is None:
            ts = np.array(np_data[:, 0])
            vertical = np.array(np_data[:, 1])
            mediolateral = np.array(np_data[:, 2])
            anteroposterior = np.array(np_data[:, 3])
        if position == 'truncate':
            csv_data = pd.read_csv(os.path.join("dataset/truncate", label + '.csv'))
            np_data = np.array(csv_data)
            ts = np.array(np_data[:, 0])
            vertical = np.array(np_data[:, 1])
            mediolateral = np.array(np_data[:, 2])
            anteroposterior = np.array(np_data[:, 3])
            label = 'truncate' + label

        elif position == 'first':
            if label in high_sample_rate:
                sample_number = 300
            ts = np.array(np_data[0:sample_number, 0])
            vertical = np.array(np_data[0:sample_number, 1])
            mediolateral = np.array(np_data[0:sample_number, 2])
            anteroposterior = np.array(np_data[0:sample_number, 3])

            # if label == '990017':
            #     ts = np.array(np_data[50:sample_number+50, 0])
            #     vertical = np.array(np_data[50:sample_number+50, 1])
            #     mediolateral = np.array(np_data[50:sample_number+50, 2])
            #     anteroposterior = np.array(np_data[50:sample_number+50, 3])

            label = 'first' + str(sample_number) + 'th' + label
        elif position == 'last':
            if label in high_sample_rate:
                sample_number = 300
            ts = np.array(np_data[-sample_number:-1, 0])
            vertical = np.array(np_data[-sample_number:-1, 1])
            mediolateral = np.array(np_data[-sample_number:-1, 2])
            anteroposterior = np.array(np_data[-sample_number:-1, 3])

            # if label == '990018':
            #     ts = np.array(np_data[-(sample_number+50):-50, 0])
            #     vertical = np.array(np_data[-(sample_number+50):-50, 1])
            #     mediolateral = np.array(np_data[-(sample_number+50):-50, 2])
            #     anteroposterior = np.array(np_data[-(sample_number+50):-50, 3])

            label = 'last' + str(sample_number) + 'th' + label
        else:
            raise Exception("Sorry, position wrong")
        fig, ax = plt.subplots(figsize=(10, 6), dpi=400)
        ax.plot(ts, vertical, label='vertical')
        ax.plot(ts, mediolateral, label='mediolateral')
        ax.plot(ts, anteroposterior, label='anteroposterior')

        ax.set_xlabel("record: " + label)
        ax.set_ylabel("sensor data")
        ax.legend()
        plt.savefig(os.path.join('./visualize/vma', label))


visualize_vma()
# time_intervals_checking()
