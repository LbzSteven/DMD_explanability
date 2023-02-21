import numpy as np
import torch


def model_test(model, trainLoader, testLoader, device, GET_OUTPUT_PROB=False):
    window_correct_test = 0
    window_total_test = 0
    window_correct_train = 0
    window_total_train = 0
    conf_total_test = 0
    conf_total_train = 0
    how_many_batch = 0
    model.eval()
    output_value = []
    output_values = None
    for i, sample in enumerate(trainLoader, 0):
        labels, data = sample['label'].to(device).to(torch.int64), sample['data'].to(device).float()

        outputs = model(data)
        # conf_total_train += torch.mean(outputs[:, labels[]])#SHAHBAZ
        _, predicted = torch.max(outputs.data, 1)
        window_total_train += labels.size(0)

        window_correct_train += (predicted == labels).sum().item()
    for i, sample_test in enumerate(testLoader, 0):
        labels_test, data_test = sample_test['label'].to(device).to(torch.int64), sample_test['data'].to(device).float()
        outputs = model(data_test)
        how_many_batch += 1
        conf_total_test += torch.mean(outputs[:, labels_test[0]])  # SHAHBAZ
        # print(outputs)
        _, predicted_test = torch.max(outputs.data, 1)
        window_total_test += labels_test.size(0)
        window_correct_test += (predicted_test == labels_test).sum().item()
        if GET_OUTPUT_PROB:
            output_value.append(outputs.cpu().detach().numpy())
    if GET_OUTPUT_PROB:
        output_values = np.concatenate(output_value)
    correct_percentage_train = window_correct_train / window_total_train
    correct_percentage_test = window_correct_test / window_total_test
    conf_total_test = conf_total_test / how_many_batch

    return correct_percentage_train, correct_percentage_test, output_values, conf_total_test