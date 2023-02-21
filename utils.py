import os
import matplotlib.pyplot as plt
import numpy as np


def normalize_oper(window_data_train, window_data_test):
    if window_data_train.ndim == 2:
        t_train = window_data_train[:, np.newaxis]
        t_test = window_data_test[:, np.newaxis]
    else:
        t_train = np.moveaxis(window_data_train, 1, 2)
        t_test = np.moveaxis(window_data_test, 1, 2)
    if window_data_train.shape[2] == 1:
        max_3_axis = np.max(t_train.reshape(-1, window_data_train.shape[2]))
        min_3_axis = np.min(t_train.reshape(-1, window_data_train.shape[2]))
    else:
        max_3_axis = np.max(t_train.reshape(-1, window_data_train.shape[2]), axis=0)
        min_3_axis = np.min(t_train.reshape(-1, window_data_train.shape[2]), axis=0)
    for i in range(window_data_train.shape[1]):
        t_train[:, :, i] = (t_train[:, :, i] - min_3_axis[i]) / (max_3_axis[i] - min_3_axis[i])
        t_test[:, :, i] = (t_test[:, :, i] - min_3_axis[i]) / (max_3_axis[i] - min_3_axis[i])
        # t_train[:, :, i] = (t_train[:, :, i] + 3) / (3 + 3)
    window_data_train = np.moveaxis(t_train, 1, 2)
    window_data_test = np.moveaxis(t_test, 1, 2)
    return window_data_train, window_data_test


def save_img_epoch_acc(train_acc_epoch, test_acc_epoch, loss_epoch, EPOCH, save_dir, number):
    # plotting
    save_dir = os.path.join(save_dir, 'epoch_acc')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.figure(figsize=(18, 15))
    ax = plt.subplot(311)
    ax.plot(range(EPOCH + 1), train_acc_epoch, label='train_acc_epoch')
    ax.set_xlabel('epoch')
    ax.set_ylabel('train_acc_epoch')
    ax.grid()
    ax.legend()

    ax = plt.subplot(312)
    ax.plot(range(EPOCH + 1), test_acc_epoch, label='test_acc_epoch')
    ax.set_xlabel('epoch')
    ax.set_ylabel('test_acc_epoch')
    ax.grid()
    ax.legend()

    ax = plt.subplot(313)
    ax.plot(range(EPOCH + 1), loss_epoch, label='loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss_epoch')
    ax.grid()
    ax.legend()

    plt.savefig(os.path.join(save_dir, number))
    plt.close()


def save_img_prob_epoch(epochs, output_values, save_dir, number):
    # np.savetxt(os.path.join(save_dir, str(fold_i + 1) + '_' + str(epochs + 1)),output_values,delimiter=',')
    plt.figure(figsize=(18, 15))
    plt.scatter(range(output_values.shape[0]), output_values[:, 0], linestyle='solid', label='TD')
    plt.scatter(range(output_values.shape[0]), output_values[:, 1], linestyle='solid', label='DMD')
    plt.legend()
    plt.savefig(os.path.join(save_dir, number + '_' + str(epochs + 1)))
    plt.close()
