import os

import pywt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

t = np.linspace(-1, 1, 200, endpoint=False)

vma = ['vertical', 'mediolateral', 'anteroposterior']

# wavlist = pywt.wavelist(kind='continuous')
wavlist = ['cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5', 'cgau6', 'cgau7', 'cgau8', 'cmor', 'fbsp', 'gaus1', 'gaus2',
           'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8', 'mexh', 'morl', 'shan']

DMD_group_number = ['990012', '990015', '990016', '990023008', '990023010', '990023015']
TD_group_number = ['990014', '990017', '990018', '990023003', '990023011', '990023014']
all_group_number = ['990012', '990014', '990015', '990016', '990017', '990018', '990023008', '990023010', '990023015',
                    '990023003', '990023011', '990023014']
low_sample_rate = ['990012', '990014', '990015', '990016', '990017', '990018']
high_sample_rate = ['990023003', '990023008', '990023010', '990023011', '990023014', '990023015']


def example_WT():
    sig = np.cos(2 * np.pi * 8 * t) + np.real(np.exp(-7 * (t - 0.4) ** 2) * np.exp(1j * 2 * np.pi * 2 * (t - 0.4)))
    # 1/7
    plt.plot(t, sig)
    plt.show()

    widths = np.arange(1, 51)

    dt = 0.01  # 100 Hz sampling
    frequency_interest = pywt.scale2frequency('gaus8', np.arange(1, 51)) / dt
    print(frequency_interest)

    cwtmatr, freqs = pywt.cwt(sig, widths, 'gaus8')
    print(cwtmatr.dtype)
    plt.imshow(cwtmatr, extent=[-1, 1, 1, 51], cmap='PRGn', aspect='auto',
               vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    plt.show()

    wavlist = pywt.wavelist(kind='continuous')
    print(wavlist)
    # wavelet = pywt.ContinuousWavelet('mexh')


def WT_DMD(group_number=None, method='mexh', widths=31):
    if group_number is None:
        group_number = all_group_number

    widths = np.arange(1, widths)

    for label in group_number:
        csv_data = pd.read_csv(os.path.join("dataset/vma", label + '.csv'))
        np_data = np.array(csv_data)
        ts = np.array(np_data[:, 0])

        vertical = np.array(np_data[:, 1])
        mediolateral = np.array(np_data[:, 2])
        anteroposterior = np.array(np_data[:, 3])

        for i in range(1, 4):
            current_signal = np_data[:, i]
            if label in high_sample_rate:
                current_signal = current_signal[0:current_signal.size:3]
            cwtmatr, freqs = pywt.cwt(current_signal, widths, method)
            direction = vma[i - 1]
            np.savetxt(os.path.join('visualize/WT', method + '_' + label + '_' + direction), cwtmatr, delimiter=',')

        # dt = 0.01  # 100 Hz sampling
        # frequency_interest = pywt.scale2frequency('mexh', np.arange(1, 31)) / dt
        # print(frequency_interest)
        # print(pywt.scale2frequency('mexh', 25) / dt)

    frequency_interest = pywt.scale2frequency(method, widths) * 33
    print(frequency_interest)
def WT_image_by_people(group_number=None, method='mexh', widths=31):
    if group_number is None:
        group_number = all_group_number
    plt.figure(1, figsize=(15, 15))
    result_dir = 'visualize/WT'
    save_dir = 'visualize/WT_image'
    for label in group_number:
        csv_data = pd.read_csv(os.path.join("dataset", label + '.csv'))
        np_data = np.array(csv_data)
        ts = np.array(np_data[:, 0])
        if label in high_sample_rate:
            ts = ts[0: ts.size:3]
        if label in DMD_group_number:
            DMD_maker = 'DMD'
        else:
            DMD_maker = 'TD'
        for i in range(1, 4):
            direction = vma[i - 1]
            cwtmatr = np.loadtxt(os.path.join(result_dir, method + '_' + label + '_' + direction), delimiter=',')

            ax = plt.subplot(310 + i)
            ax.imshow(cwtmatr,  cmap='PRGn', aspect='auto',
                      vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
            ax.set_title(direction)
            # ax.set_xlabel('Translation')
            ax.set_ylabel('Scale')
            # ax.legend()
        plt.suptitle('Wavelet transform ' + method + '_' + label + '_' + DMD_maker, fontsize=20)
        # plt.xlabel('Frequency(Hz)', fontsize=20)
        plt.savefig(os.path.join(save_dir, method + '_' + label))


dt = 0.01  # 100 Hz sampling
frequency_interest = pywt.scale2frequency('gaus8', np.arange(1, 31)) * 33
print(frequency_interest)

# WT_DMD(method='morl')
# WT_image_by_people(method='mexh')
# example_WT()
