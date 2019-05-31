import csv
import sys
import time
import pandas as pd

from train import Model

test_set    = pd.read_csv('./data/test.csv')
test_tweets = pd.read_csv('./data/test-tweets.txt', sep="\t", header=None)

test_set.drop('id', axis = 1, inplace = True)

class Predictor:

    model = Model()


    def predict(self):

        clf_naive_bayes = self.model.naive_bayes()
        test_pred = clf_naive_bayes.predict(test_set)

        print(test_pred)

        # test_pred = clf_model.fit(X, y).predict(test_set)
        #         with open('./test_labels_bayes.txt', 'w') as f:
        #             for (id, label) in zip(test_tweets[0], test_pred):
        #                 f.write(str(id) + '\t' + str(label) + '\r\n')
