import time
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression as LR
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB as NB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DT

from run_iBCM import iBCM, iBCM_verify



def evaluate_classifiers(train_feature, train_label, test_feature, test_label):
    classifiers = {
        "RandomForest": RF(),
        "SVM": SVC(),
        "NaiveBayes": BernoulliNB(),
        "KNN": KNeighborsClassifier(),
        "DecisionTree": DT()
    }
    accuracies = {name: [] for name in classifiers.keys()}

    for _ in range(1):
        for name, clf in classifiers.items():
            clf.fit(train_feature, train_label)
            predictions = clf.predict(test_feature)
            accuracies[name].append(accuracy_score(test_label, predictions))

    results = {name: (np.mean(acc), np.std(acc)) for name, acc in accuracies.items()}
    return results
def run_iBCM(dataset, support,times):
    # Read files
    trace_file = open('./datasets/'+dataset+'.dat', 'r')
    label_file = open('./datasets/'+dataset+'.lab', 'r')

    #####################################
    # Store information and create folds
    traces = []
    label_list = []
    for trace, label in zip(trace_file, label_file):
        traces.append(trace)
        label_list.append(label.replace('\n', ''))
    # print('Number of traces: ', traces)

    label_set = set(label_list)
    no_labels = len(label_set)
    print('#labels:', no_labels)

    skf = KFold(no_folds, shuffle=True)

    ##########################
    # Apply iBCM on all folds
    fold_train_results = []
    fold_test_results = []

    acc_sum = 0
    feat_sum = 0
    auc_sum = 0
    # time 
    start_time = time.time()

    for fold, (train_index, test_index) in enumerate(skf.split(traces, label_list)):
        print('\nFold ', (fold+1), '/', no_folds)
        training_points = []
        test_points = []
        training_labels = []
        test_labels = []

        for i in train_index:
            training_points.append(traces[i])
            training_labels.append(label_list[i])

        filename_train = dataset + '_training_fold_' + str(fold) + '_support_'
        filename_train += str(support) + '.csv'

        final_constraints = iBCM(filename_train, training_points,
                                 training_labels, reduce_feature_space, support, no_win)

        # Label training data
        iBCM_verify(filename_train, training_points,
                    training_labels, final_constraints, no_win)

        fold_train_results.append(pd.read_csv(filename_train))

        filename_test = dataset + '_test_fold_' + str(fold) + '_support_'
        filename_test += str(support) + '.csv'

        for i in test_index:
            test_points.append(traces[i])
            test_labels.append(label_list[i])

        # Label test data
        iBCM_verify(filename_test, test_points,
                    test_labels, final_constraints, no_win)
        fold_test_results.append(pd.read_csv(filename_test))

        os.remove(filename_train)
        os.remove(filename_test)

    #######################
    # Apply classification


    res_acc = []
    for i in range(0, no_folds):
        #                print('Fold:',i)
        training = fold_train_results[i]
        test = fold_test_results[i]
        y_train = training['label']
        X_train = training.drop(['label'], axis=1)
        y_test = test['label']
        X_test = test.drop(['label'], axis=1)
        # print('#Features:', len(X_train.columns))
        if len(X_train.columns) < 2:
            print('No features for fold', i)
            continue
        acc = evaluate_classifiers(X_train,y_train,X_test,y_test)
        res_acc.append(acc)

    return res_acc


# Start program and enter parameters
no_folds = 5
no_win = 1
reduce_feature_space = False

write_results = True

for dataset in  ['aslbu', 'auslan2', 'context', 'epitope', 'gene', 'pioneer', 'question', 'reuters', 'robot', 'skating', 'unix']:
    print('\nDataset:', dataset)
    support = 0.1
    total_accuracy = []
    for i in range(10):
        print('\nSupport level:', i)
        total_accuracy.append(run_iBCM(dataset, support,i))
            # Calculate mean and std for each classifier
    classifier_names = total_accuracy[0][0].keys()
    results = {name: [] for name in classifier_names}
    for name in classifier_names:
        acc_10 = []
        for i in range(10):
            acc_5_mean = 0
            for j in range(5):
                acc_5_mean += total_accuracy[i][j][name][0]
            acc_10.append(acc_5_mean/5)
        results[name] = [np.mean(acc_10), np.std(acc_10)]
    results_df = pd.DataFrame(results, index=['mean', 'std']).T
    folder = 'iBCM_results_1'
    result_filename = f'result_{dataset}.csv'
    results_df.to_csv(os.path.join(folder, result_filename))

    print("\nMean and Std of 10 experiments:")
    print(results_df)
