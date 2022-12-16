import pywt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

t = np.linspace(-1, 1, 200, endpoint=False)

vma = ['vertical', 'mediolateral', 'anteroposterior']

# wavlist = pywt.wavelist(kind='continuous')
wavlist = ['cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5', 'cgau6', 'cgau7', 'cgau8', 'cmor', 'fbsp', 'gaus1', 'gaus2',
           'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8', 'mexh', 'morl', 'shan']
print(wavlist)


def example_WT():
    sig = np.cos(2 * np.pi * 7 * t) + np.real(np.exp(-7 * (t - 0.4) ** 2) * np.exp(1j * 2 * np.pi * 2 * (t - 0.4)))

    plt.plot(t, sig)
    plt.show()

    widths = np.arange(1, 31)

    dt = 0.01  # 100 Hz sampling
    frequency_interest = pywt.scale2frequency('mexh', np.arange(1, 50)) / dt
    print(frequency_interest)

    cwtmatr, freqs = pywt.cwt(sig, widths, 'mexh')
    plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
               vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    plt.show()

    wavlist = pywt.wavelist(kind='continuous')
    print(wavlist)
    print(freqs)
    # wavelet = pywt.ContinuousWavelet('mexh')


def example_WT_DMD():
    data_path = 'dataset/vma/990023008.csv'
    csv_data = pd.read_csv(data_path)
    np_data = np.array(csv_data)
    ts = np.array(np_data[:, 0])

    vertical = np.array(np_data[:, 1])
    mediolateral = np.array(np_data[:, 2])
    anteroposterior = np.array(np_data[:, 3])

    widths = np.arange(1, 31)
    plt.figure(1, figsize=(15, 15))
    for i in range(1, 4):
        cwtmatr, freqs = pywt.cwt(np_data[:, i], widths, 'mexh')
        name = vma[i - 1]
        ax = plt.subplot(310+i)
        ax.imshow(cwtmatr, extent=[ts[0], ts[-1], 1, 31], cmap='PRGn', aspect='auto',
                   vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
        ax.set_title(name)
        # ax.set_xlabel('Translation')
        ax.set_ylabel('Scale')
        ax.legend()
    plt.show()

    dt = 0.01  # 100 Hz sampling
    frequency_interest = pywt.scale2frequency('mexh', np.arange(1, 31)) / dt
    print(frequency_interest)
    print(pywt.scale2frequency('mexh', 25) / dt)
example_WT_DMD()
