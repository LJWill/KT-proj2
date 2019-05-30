import sys
from model import Model


def main(train_method):
    
    clf_model = Model()
    clf_model.data_preprocessing()
        # clf_model.data_analysis(clf_model.eval_label[1])
        # clf_model.data_analysis(clf_model.train_label[1])

    if train_method == 'nb':     clf_model.naive_bayes()
    elif train_method == 'svm':  clf_model.svm()
    elif train_method == 'dt':   clf_model.decision_tree()
    elif train_method == 'rf':   clf_model.random_forest()
    elif train_method == 'knn':  clf_model.knn()
    else: print('No methods found')

if __name__ == "__main__":
    main(sys.argv[1])

