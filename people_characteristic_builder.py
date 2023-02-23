import csv
import os

import pandas as pd
import numpy as np
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
        self.average = None
        self.voting = None

    def get_datasets(self):
        for path in people30_dataset_path_list:
            data_path = os.path.join(path, self.N + '.csv')
            if os.path.exists(data_path):
                self.paths.append(data_path)
            else:
                self.paths.append(None)
            self.dataset_marker.append(path.split('/')[-1])
            self.predictions.append(np.nan) # convenient for computing

    def get_predictions(self):
        return self.predictions

    def set_prediction(self, i, pred):
        self.predictions[i] = pred

    def average_results(self):
        predictions = np.array(self.predictions)
        self.average = np.nanmean(predictions)

    def voting_results(self):
        predictions = np.array(self.predictions)
        per = np.sum(np.logical_and(predictions is not None, predictions > 0.5)) / np.count_nonzero(predictions is not None)
        if per > 0.5:
            self.voting = 'C'
        else:
            self.voting = 'W'

    def prediction_to_csv(self, file=r'major_voting.csv'):
        with open(file, mode='a', newline='', encoding='utf8') as cfa:
            wf = csv.writer(cfa)
            data = [self.ID, self.N] + self.predictions + [self.voting, self.average]
            wf.writerow(data)


df = pd.read_csv('30people_labels.csv')
people_list = []

for i in range(df.shape[0]):
    values = df.iloc[i, :]
    p = person(values['ID'], values['N'], values['Date'], values['Case'],
               values['Age'], values['Weight'], values['Height'], values['NSAA'])
    people_list.append(p)

p = people_list[0]
pred = [0.5, 0.6, 0.7, 0.8, np.nan, 0.0, 0.05]
for i in range(len(pred)):
    p.set_prediction(i, pred)
p.voting_results()
p.average_results()
print(p.voting, p.average)
# sanity check
# people_lists[0].set_prediction(i=1, pred=0.5)
# people_lists[0].prediction_to_csv()
# print(len(people_lists))
# with open(r'major_voting.csv', mode='a', newline='', encoding='utf8') as cfa:
#     wf = csv.writer(cfa)
#     data = ['ID', 'N'] + [i.split('/')[-1] for i in people_30_path_list]
#     wf.writerow(data)
