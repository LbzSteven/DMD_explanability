import csv


def csv_writer(epoch, WT_step, WT_size, model_para, acc):
    with open(r'acc.csv', mode='a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        data = [epoch, WT_step, WT_size, model_para, acc]
        wf.writerow(data)


csv_writer(10000, 33, 33, 'B_128_lr_0.001', 0.50)
