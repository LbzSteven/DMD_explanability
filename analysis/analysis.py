import numpy as np
import matplotlib.pyplot as plt
import os
from people_characteristic_builder import people_list

wrong_list = ['23031', '23033', '23014', '230041']
# plot the probability
direction = '../save_result/' \
            'resnet48_avg_softmax_100_meter160_step5_Norm_False_Epoch_acc_prob_out_02_18_01_44/origin_output_prob'
print(os.path.exists(direction))
path = os.path.join(direction, wrong_list[0] + '.npy')
print(os.path.exists(path))
t = np.load(path)
output_values = t[-1, :, :]
plt.figure(figsize=(18, 15))
# plt.scatter(range(output_values.shape[0]), output_values[:, 0], linestyle='solid', label='TD')
plt.scatter(range(output_values.shape[0]), output_values[:, 1], linestyle='solid')
plt.plot(range(output_values.shape[0]), output_values[:, 1], linestyle='solid', label='DMD')
plt.legend()
plt.show()
plt.close()
# plot the epoch_acc

# plot the FFT
