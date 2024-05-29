import os
import numpy as np
import torch
from sklearn.model_selection import KFold
from HammingEncoder import HammingEncoder
from data_generator import datainput
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, naive_bayes, neighbors, tree
from sklearn.metrics import accuracy_score

def k_folder_split(db, data_label, k=5):
    kf = KFold(n_splits=k, shuffle=True)
    total_dataset = kf.split(db, data_label)
    x_train_list = []
    x_test_list = []
    y_train_list = []
    y_test_list = []

    for _ in range(k):
        train_index, test_index = next(total_dataset)
        x_train = [db[i] for i in train_index]
        x_test = [db[i] for i in test_index]
        y_train, y_test = np.array(data_label)[train_index], np.array(data_label)[test_index]

        x_train_list.append(x_train)
        x_test_list.append(x_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)

    return x_train_list, x_test_list, y_train_list, y_test_list



def trainning(n_epochs, patience, times, batchsize, dataset, kmer_length, patten_number):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    db, data_label, itemset, max_sequence_length = datainput('dataset/{}.txt'.format(dataset))

    class_dict = {label: idx for idx, label in enumerate(set(data_label))}

    x_train_list, x_test_list, y_train_list, y_test_list = k_folder_split(db, data_label)

    for i in range(times):
        model = HammingEncoder(x_train_list[i], y_train_list[i], gap_constrain=kmer_length, label_number=len(class_dict),
                               Preset_set_pattern_num=patten_number).to(device)
        model.fit(n_epochs, patience, batchsize, kmer_length, patten_number, device)
        test_feature = model.transform(x_test_list[i])
        train_feature = model.transform(x_train_list[i])
        
        

def k_folder_split(db, data_label, k=5):
    kf = KFold(n_splits=k, shuffle=True)
    total_dataset = kf.split(db, data_label)
    x_train_list = []
    x_test_list = []
    y_train_list = []
    y_test_list = []

    for _ in range(k):
        train_index, test_index = next(total_dataset)
        x_train = [db[i] for i in train_index]
        x_test = [db[i] for i in test_index]
        y_train, y_test = np.array(data_label)[train_index], np.array(data_label)[test_index]

        x_train_list.append(x_train)
        x_test_list.append(x_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)

    return x_train_list, x_test_list, y_train_list, y_test_list


def evaluate_classifiers(train_feature, train_label, test_feature, test_label):
    classifiers = {
        "RandomForest": RandomForestClassifier(),
        "SVM": svm.SVC(),
        "NaiveBayes": naive_bayes.BernoulliNB(),
        "KNN": neighbors.KNeighborsClassifier(),
        "DecisionTree": tree.DecisionTreeClassifier()
    }
    accuracies = {name: [] for name in classifiers.keys()}

    for _ in range(1):
        for name, clf in classifiers.items():
            clf.fit(train_feature, train_label)
            predictions = clf.predict(test_feature)
            accuracies[name].append(accuracy_score(test_label, predictions))

    results = {name: (np.mean(acc), np.std(acc)) for name, acc in accuracies.items()}
    return results

def trainning(n_epochs, patience, times, batchsize, dataset, kmer_length, patten_number):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    db, data_label, itemset, max_sequence_length = datainput(f'dataset/{dataset}.txt')
    x_train_list, x_test_list, y_train_list, y_test_list = k_folder_split(db, data_label)
    all_res = []
    for i in range(times):
        model = HammingEncoder(
            x_train_list[i], y_train_list[i], gap_constrain=kmer_length,
            label_number=len(set(data_label)), Preset_set_pattern_num=patten_number
        ).to(device)
        model.fit(n_epochs, patience, batchsize, kmer_length, patten_number, device)
        train_feature = model.transform(x_train_list[i])
        test_feature = model.transform(x_test_list[i])

        results = evaluate_classifiers(train_feature, y_train_list[i], test_feature, y_test_list[i])
        print(f'Experiment {i + 1}: {results}')
        all_res.append(results)
    # return the average of the results
    
    return all_res