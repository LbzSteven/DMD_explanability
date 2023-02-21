import csv


def csv_writer_acc_count_wpl(word_marker, correct_percentages, correct_person_count, wrong_person_list, time):
    with open(r'acc_count_wpl.csv', mode='a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        data = [word_marker, correct_percentages, correct_person_count, wrong_person_list, time]
        wf.writerow(data)


def csv_writer(epoch, WT_step, WT_size, model_para, acc):
    with open(r'acc.csv', mode='a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        data = [epoch, WT_step, WT_size, model_para, acc]
        wf.writerow(data)


# csv_writer(10000, 33, 33, 'B_128_lr_0.001', 0.50)
# csv_writer_acc_count_wpl('description', [0.55, 0.55], 15, ['15', '16'], 'time')
