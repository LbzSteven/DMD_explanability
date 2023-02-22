import csv
import os

import pandas as pd
from constants import people30_dataset_path_list


# Define the data structure to hold information about a person

class person:

    def __init__(self, ID, N, Date, Case, Age, Weight, Height, NSAA):
        self.ID = ID
        self.N = str(N)
        self.Date = Date
        self.Case = Case
        self.Age = Age
        self.Weight = Weight
        self.Height = Height
        self.NSAA = NSAA

        self.predictions = []
        self.dataset_marker = []
        self.paths = []
        self.get_datasets()

    def get_datasets(self):
        for path in people30_dataset_path_list:
            data_path = os.path.join(path, self.N + '.csv')
            if os.path.exists(data_path):
                self.paths.append(data_path)
            else:
                self.paths.append(None)
            self.dataset_marker.append(path.split('/')[-1])
            self.predictions.append(None)

    def get_predictions(self):
        return self.predictions

    def set_prediction(self, i, pred):
        self.predictions[i] = pred

    def prediction_to_csv(self, file=r'major_voting.csv'):
        with open(file, mode='a', newline='', encoding='utf8') as cfa:
            wf = csv.writer(cfa)
            data = [self.ID, self.N] + self.predictions
            wf.writerow(data)


df = pd.read_csv('30people_labels.csv')
people_list = []

for i in range(df.shape[0]):
    values = df.iloc[i, :]
    p = person(values['ID'], values['N'], values['Date'], values['Case'],
               values['Age'], values['Weight'], values['Height'], values['NSAA'])
    people_list.append(p)

# sanity check
# people_lists[0].set_prediction(i=1, pred=0.5)
# people_lists[0].prediction_to_csv()
# print(len(people_lists))
# with open(r'major_voting.csv', mode='a', newline='', encoding='utf8') as cfa:
#     wf = csv.writer(cfa)
#     data = ['ID', 'N'] + [i.split('/')[-1] for i in people_30_path_list]
#     wf.writerow(data)
