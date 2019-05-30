import csv
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import naive_bayes
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn import ensemble
from sklearn import metrics



class Model:

    train_set = pd.read_csv('./data/train.csv')
    train_label = pd.read_csv('./data/train-labels.txt', sep="\t", header=None)
    eval_set = pd.read_csv('./data/eval.csv')
    eval_label = pd.read_csv('./data/eval-labels.txt', sep="\t", header=None)
    
    def data_analysis(self, label_list):
        no_positive = 0
        no_negative = 0
        no_neutral  = 0

        for item in label_list:
            if item == 'positive':
                no_positive += 1
            elif item == 'negative':
                no_negative += 1
            else:
                no_neutral  += 1

        print(no_positive, no_negative, no_neutral)

        labels = 'positive', 'negative', 'neutral'
        sizes = [no_positive, no_negative, no_neutral]
        colors = ['gold', 'yellowgreen', 'lightcoral'] #, 'lightskyblue']
        explode = (0.1, 0, 0)  # explode 1st slice
        
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.2f%%', shadow=True, startangle=140)
        
        plt.axis('equal')
        plt.show()


    def naive_bayes(self):
        clf_model = naive_bayes.MultinomialNB()
        # clf_model = naive_bayes.GaussianNB()

        eval_pred = clf_model.fit(self.train_set, self.train_label[1]).predict(self.eval_set)

        self.output_result(eval_pred)

    def svm(self):
        # clf_model = svm.SVC(kernel='linear')
        clf_model = svm.SVC(kernel='rbf')
        eval_pred = clf_model.fit(self.train_set, self.train_label[1]).predict(self.eval_set)

        self.output_result(eval_pred)


    def decision_tree(self):
        clf_model = tree.DecisionTreeClassifier(random_state=0, max_depth=11)
        eval_pred = clf_model.fit(self.train_set, self.train_label[1]).predict(self.eval_set)

        self.output_result(eval_pred)


    def random_forest(self):
        clf_model =  RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=15, random_state=0)
        eval_pred = clf_model.fit(self.train_set, self.train_label[1]).predict(self.eval_set)

        self.output_result(eval_pred)

    def knn(self):
        clf_model = KNeighborsClassifier(n_neighbors = 3, algorithm="auto", leaf_size=30)
        eval_pred = clf_model.fit(self.train_set, self.train_label[1]).predict(self.eval_set)

        self.output_result(eval_pred)





    def output_result(self, eval_pred):
        target_names = ['negative', 'neutral', 'positive']
        result = classification_report(self.eval_label[1], eval_pred, target_names=target_names)

        print(result)

        no_correct = (eval_pred == self.eval_label[1]).sum()
        percent = no_correct / len(self.eval_label)

        print('Number of correct predict: ', no_correct, 'Percentage: ', percent, '\n')


    def is_id_match(self, list1, list2):
        if len(list1) != len(list2): return False

        result = True
        for a, b in zip(list1, list2):
            if (len(a) != len( str(b) )):
                if str(b) not in a: result = False
            else:
                if str(b) != a: result = False

        return result


    def data_preprocessing(self):
            
        if self.is_id_match(self.train_set['id'], self.train_label[0]):
            self.train_set.drop('id', axis = 1, inplace = True)
            self.train_label.drop(0, axis = 1, inplace = True)

        if self.is_id_match(self.eval_set['id'], self.eval_label[0]):
            self.eval_set.drop('id', axis = 1, inplace = True)
            self.eval_label.drop(0, axis = 1, inplace = True)