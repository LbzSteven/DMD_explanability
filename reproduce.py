import numpy as np
import torch
from torch import nn

from window import window_oper
from CNN_torch import CNN_DMD
from dataset import DMDDataset
from result_record import csv_writer
import torchvision.transforms as transforms
import torch.optim as optim

DMD_group_number = ['990012', '990015', '990016', '990023008', '990023010', '990023015']
TD_group_number = ['990014', '990017', '990018', '990023003', '990023011', '990023014']
all_group_number = ['990012', '990014', '990015', '990016', '990017', '990018', '990023008', '990023010', '990023015',
                    '990023003', '990023011', '990023014']
low_sample_rate = ['990012', '990014', '990015', '990016', '990017', '990018']
high_sample_rate = ['990023003', '990023008', '990023010', '990023011', '990023014', '990023015']

# WINDOW_SIZE = 33
# WINDOW_STEP = 33
LEARN_RATE = 0.001
BATCH_SIZE = 128
# EPOCH = 100000
NUM_WORKERS = 0

# 12 fold training and test
# transforms = transforms.Compose(
#     [
#         transforms.ToTensor(),
#     ]
# )
transforms = None

j = 0
for EPOCH in [100, 500, 1000, 5000, 10000, 50000]:
    for WINDOW_SIZE in [30, 33, 90, 100]:
        for WINDOW_STEP in [WINDOW_SIZE, int(WINDOW_SIZE / 2), int(WINDOW_SIZE / 3)]:
            correct = 0
            total = len(all_group_number)
            paitent_makers, window_labels, window_data = window_oper(all_group_number, WINDOW_SIZE, WINDOW_STEP)
            window_labels = np.array(window_labels)
            window_data = np.array(window_data)
            print('current: EPOCH %d W_SIZE %d W_STEP %d' % (EPOCH, WINDOW_SIZE, WINDOW_STEP))
            for number in all_group_number:
                j += 1
                testing_idx = [i for i, x in enumerate(paitent_makers) if x == number]
                training_idx = [i for i, x in enumerate(paitent_makers) if x != number]

                testing_labels = window_labels[testing_idx]
                testing_data = window_data[testing_idx, :]
                training_labels = window_labels[training_idx]
                training_data = window_data[training_idx, :]

                trainset = DMDDataset(training_labels, training_data, transforms)
                testset = DMDDataset(testing_labels, testing_data, transforms)

                trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                                          shuffle=True)
                testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                                         shuffle=False)

                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

                net = CNN_DMD(WINDOW_SIZE).float().to(device)
                optimizer = optim.Adam(net.parameters())
                loss_function = nn.CrossEntropyLoss()
                # print('patient: ', number, ' train start')
                for epoch in range(EPOCH):
                    running_loss = 0
                    for i, (sample) in enumerate(trainloader, 0):
                        # get the inputs

                        labels, data = sample['label'].to(device).to(torch.int64), sample['data'].to(device).float()

                        optimizer.zero_grad()
                        outputs = net(data)

                        loss = loss_function(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                    # print(running_loss)
                window_correct = 0
                window_total = 0
                # print('patient: ', number, ' train finished')
                with torch.no_grad():
                    for i, sample in enumerate(testloader, 0):
                        labels, data = sample['label'].to(device).to(torch.int64), sample['data'].to(device).float()

                        outputs = net(data)
                        _, predicted = torch.max(outputs.data, 1)
                        window_total += labels.size(0)

                        window_correct += (predicted == labels).sum().item()
                        # print(predicted)
                        # print(labels)
                        # print(window_correct)
                        # print(window_total)
                correct_percentage = window_correct / window_total

                if correct_percentage > 0.5:
                    correct += 1
                    # print(' patient: %s predict correct, percentage:%.2f,loss:%.5f' % (number,  correct_percentage, running_loss))
                # else:
                # print(' patient: %s predict wrong, percentage:%.2f,loss:%.5f' % (number,  correct_percentage, running_loss))

                # print('total correct percentage:', correct / total)
            csv_writer(EPOCH, WINDOW_STEP, WINDOW_SIZE, 'bs_' + str(BATCH_SIZE) + '_lr_' + str(LEARN_RATE),
                       (correct / total))
