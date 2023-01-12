import csv


def csv_writer(model_para, acc):
    with open(r'acc.csv', mode='a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        data = [model_para, acc]
        wf.writerow(data)


csv_writer('CNN_B_128_S_33_S33', 0.50)
