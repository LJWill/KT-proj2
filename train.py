import csv
import sys
import time
import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from sklearn import naive_bayes
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn import ensemble
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from nltk.corpus import stopwords



# for tunning hyperparameters
is_tunning_mode = False

# for predicting test set
is_predicting_mode = False

class Model:

    train_set   = pd.read_csv('./data/train.csv')
    train_label = pd.read_csv('./data/train-labels.txt', sep="\t", header=None)
    eval_set    = pd.read_csv('./data/eval.csv')
    eval_label  = pd.read_csv('./data/eval-labels.txt', sep="\t", header=None)

    def naive_bayes(self, X=train_set, y=train_label[1]):
        clf_model = naive_bayes.MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
        # clf_model = naive_bayes.GaussianNB()

        eval_pred = clf_model.fit(X, y).predict(self.eval_set)
        self.output_result(eval_pred, 'Naive Bayes')

        return clf_model.fit(X,y)



    def svm(self, X=train_set, y=train_label[1]):
        if is_tunning_mode:
            cs = [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]
            gammas = [0.01, 0.1, 1, 2, 3, 'auto']
            rand_param_grid = {
                'kernel':('linear', 'rbf'), 
                'C': cs,'gamma': gammas,
                'decision_function_shape':('ovo','ovr'),
                'shrinking':(True,False)
            }
            
            grid = RandomizedSearchCV(svm.SVC(), rand_param_grid, 
                   n_iter = 100, cv = 5, verbose=2, n_jobs = -1, scoring='accuracy')
            grid.fit(X, y)
            grid.grid_scores_


            print(grid.best_params_)
            print(grid.best_score_)

        else:
            clf_model = svm.SVC()
            eval_pred = clf_model.fit(X, y).predict(self.eval_set)
            self.output_result(eval_pred, 'SVM')

            return clf_model.fit(X,y)


    def decision_tree(self, X=train_set, y=train_label[1]):
        if is_tunning_mode:
            # Maximum number of levels in tree
            max_depth = list(range(3, 30))
            # Minimum number of samples required to split a node
            min_samples_split = [2, 5, 10]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1, 2, 4]
            criterion = ['gini','entropy']

            rand_param_grid = {
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'criterion': criterion
            }
    
            grid = RandomizedSearchCV(tree.DecisionTreeClassifier(random_state=0), rand_param_grid, 
                   n_iter = 100, cv = 10, verbose=2, random_state=42, n_jobs = -1, scoring='accuracy')

            grid.fit(X, y)

            print(grid.best_params_)  # {'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': 11, 'criterion': 'gini'}
            print(grid.best_score_)   # 0.5470918345151607

        else:
            clf_model = tree.DecisionTreeClassifier(max_depth=11, criterion='gini', min_samples_leaf=1, min_samples_split=5)
            eval_pred = clf_model.fit(X, y).predict(self.eval_set)
            self.output_result(eval_pred, 'Decision Tree')

            return clf_model.fit(X,y)


    def random_forest(self, X=train_set, y=train_label[1]):
        if is_tunning_mode:
            # Number of trees in random forest
            n_estimators = list(range(20, 200, 5))
            # Maximum number of levels in tree
            max_depth = list(range(3, 45))
            # Minimum number of samples required to split a node
            min_samples_split = [2, 5, 10]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1, 2, 4]
            # Method of selecting samples for training each tree
            bootstrap = [True, False]
            criterion = ['gini','entropy']

            rand_param_grid = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap,
                'criterion': criterion
            }
    
            grid = RandomizedSearchCV(RandomForestClassifier(random_state=0), rand_param_grid, 
                   n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1, scoring='accuracy')

            grid.fit(X, y)

            print(grid.best_params_)  # {'n_estimators': 195, 'min_samples_split': 5, 'min_samples_leaf': 4, 
                                      #  'max_depth': 42, 'criterion': 'entropy', 'bootstrap': True}
            print(grid.best_score_)   # 0.5552268673598121

        else:
            clf_model =  RandomForestClassifier(n_estimators=195, min_samples_split=5, min_samples_leaf=4, 
                                               max_depth=42, criterion='entropy', bootstrap=True)
            eval_pred = clf_model.fit(X, y).predict(self.eval_set)
            self.output_result(eval_pred, 'Random Forest')

            return clf_model.fit(X,y)



    def knn(self, X=train_set, y=train_label[1]):
        if is_tunning_mode:
            n_neighbors = list(range(1,15))
            weight_options = ['uniform','distance']
            metric = ['euclidean', 'manhattan']
            algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']

            param_grid ={
                'n_neighbors': n_neighbors,
                'weights': weight_options,
                'metric': metric,
                'algorithm': algorithm
            }

            grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=3, scoring='accuracy', verbose=2)

            grid.fit(X, y)

            print(grid.best_params_)  # {'algorithm': 'ball_tree', 'metric': 'euclidean', 'n_neighbors': 13, 'weights': 'uniform'}
            print(grid.best_score_)   # 0.5132031148040197

        else:
            clf_model = KNeighborsClassifier(n_neighbors = 13, algorithm="ball_tree", metric="euclidean", weights="uniform")
            eval_pred = clf_model.fit(X, y).predict(self.eval_set)
            self.output_result(eval_pred, 'KNN')

            return clf_model.fit(X,y)





    def output_result(self, eval_pred, method_name):
        if is_predicting_mode:
            pass
        else:
            target_names = ['negative', 'neutral', 'positive']
            result = classification_report(self.eval_label[1], eval_pred, target_names=target_names)
            print('*****************************************************************')
            print(method_name + ': \n')
            print(result)

            no_correct = (eval_pred == self.eval_label[1]).sum()
            percent = no_correct / len(self.eval_label)

            print('Number of correct predict: ', no_correct, 'Accuracy: ', percent, '\n')
            print('*****************************************************************')

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