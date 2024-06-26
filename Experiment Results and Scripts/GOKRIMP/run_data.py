import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import subprocess
import timeit
import pandas as pd
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import numpy as np

def evaluate_classifiers(train_feature, train_label, test_feature, test_label):
    classifiers = {
        "RandomForest": RandomForestClassifier(),
        "SVM": SVC(),
        "NaiveBayes": BernoulliNB(),
        "KNN": KNeighborsClassifier(),
        "DecisionTree": DecisionTreeClassifier()
    }
    accuracies = {name: [] for name in classifiers.keys()}

    for _ in range(1):
        for name, clf in classifiers.items():
            clf.fit(train_feature, train_label)
            predictions = clf.predict(test_feature)
            accuracies[name].append(accuracy_score(test_label, predictions))

    results = {name: (np.mean(acc), np.std(acc)) for name, acc in accuracies.items()}
    return results


def read_patterns(file_path):
    try:
        patterns = []
        with open(file_path, 'r') as file:
            for line in file:
                sup_index = line.find('#SUP')  # 查找#SUP的位置
                if sup_index != -1:  # 如果找到了#SUP
                    numbers_str = line[:sup_index].strip()  # 取#SUP前面的部分
                    numbers_list = [num for num in numbers_str.split() if num != '-1']  # 按空格分割数字，并过滤掉-1
                    patterns.append(numbers_list)
        if len(patterns) == 0:
            return None
        return patterns
    except FileNotFoundError:
        print("File not found.")
        return None
    except Exception as e:
        print("An error occurred:", e)
        return None
    

       

def generate_the_split(dataset, no_folds=5):
    print('Handling dataset:', dataset)

    # Ensure the directory exists for storing the datasets
    if not os.path.exists('datasets'):
        os.makedirs('datasets')

    trace_file = open(f'datasets/{dataset}.dat', 'r')
    label_file = open(f'datasets/{dataset}.lab', 'r')

    traces = []
    label_list = []
    for trace, label in zip(trace_file, label_file):
        traces.append(trace)
        label_list.append(label.replace('\n', ''))

    trace_file.close()
    label_file.close()

    skf = KFold(n_splits=no_folds, shuffle=True, random_state=42)  # Added a random state for reproducibility

    for fold, (train_index, test_index) in enumerate(skf.split(traces, label_list)):
        train_traces = [traces[i] for i in train_index]
        train_labels = [label_list[i] for i in train_index]
        test_traces = [traces[i] for i in test_index]
        test_labels = [label_list[i] for i in test_index]

        # Save training and testing files for each fold
        with open(f'datasets/{dataset}_training_fold_{fold}.dat', 'w') as f:
            f.writelines(train_traces)
        with open(f'datasets/{dataset}_training_fold_{fold}.lab', 'w') as f:
            f.writelines(f"{label}\n" for label in train_labels)

        with open(f'datasets/{dataset}_testing_fold_{fold}.dat', 'w') as f:
            f.writelines(test_traces)
        with open(f'datasets/{dataset}_testing_fold_{fold}.lab', 'w') as f:
            f.writelines(f"{label}\n" for label in test_labels)

        print(f'Fold {fold} processed.')

def read_dataset(dataset, fold, type):
    """
    Reads the dataset for a specific fold and type (training or testing),
    and returns the features and labels with entries marked as '-1' removed.

    Parameters:
        dataset (str): Name of the dataset.
        fold (int): Fold number.
        type (str): Either 'training' or 'testing'.

    Returns:
        tuple: Two lists, one of features and one of labels.
    """
    data_path = f'/mnt/sda1/code/GOKRIMP/datasets/{dataset}_{type}_fold_{fold}.dat'
    label_path = f'/mnt/sda1/code/GOKRIMP/datasets/{dataset}_{type}_fold_{fold}.lab'

    # Initialize lists to hold the training data and labels
    X = []
    y = []

    with open(data_path, 'r') as data_file, open(label_path, 'r') as label_file:
        for data_line, label_line in zip(data_file, label_file):
            data_entries = data_line.replace('\n', '').split(' ')
            # Filter out '-1' entries
            data_entries = [x for x in data_entries if x != '-1'][:-1]
            label = label_line.strip()

            X.append(data_entries)
            y.append(label)

    return X, y



def pattern_feature_transform(patterns, X_train):
    # Initialize the feature matrix with zeros
    feature_x = np.zeros((len(X_train), len(patterns)))
    
    # Iterate over all patterns
    for i, pattern in enumerate(patterns):
        pattern_length = len(pattern)
        
        # Iterate over all traces
        for j, trace in enumerate(X_train):
            # Check if the pattern is a subsequence of the trace
            if is_subsequence(pattern, trace):
                feature_x[j][i] = 1
    
    return feature_x

def is_subsequence(pattern, trace):
    # Helper function to check if pattern is a subsequence of trace
    it = iter(trace)
    return all(item in it for item in pattern)

# 'auslan2', 'aslbu','pioneer','skating','question','context', 
datasets = [ 'reuters', 'unix', 'robot']
data_dict = {name: [] for name in datasets}


for dataset in datasets:
    
    total_accuracy = []
    for times in tqdm(range(10)):

        generate_the_split(dataset, no_folds = 5)
        all_res = []
        for fold in range(5):
            #############################################
            ### get MiSeRe features
            subprocess.run(
                # java -jar spmf.jar run GoKrimp test_goKrimp.dat output.txt test_goKrimp.lab
                f"java -jar spmf.jar run GoKrimp datasets/{dataset}_training_fold_{fold}.dat res datasets/{dataset}_training_fold_{fold}.lab", shell=True)
            print('mining finished')
            #############################################
            #read  MiSeRe features  res
            file_path = './res'  # 文件名是固定的
            patterns = read_patterns(file_path)
            train_X, train_y = read_dataset(dataset, fold, 'training')
            test_x, test_y = read_dataset(dataset, fold, 'testing')

            # Transform the patterns into features
            train_x = pattern_feature_transform(patterns, train_X)
            test_x = pattern_feature_transform(patterns, test_x)

            acc = evaluate_classifiers(train_x, train_y, test_x, test_y)
                
            all_res.append(acc)
                # average the result
        total_accuracy.append(all_res)
        
    # calculate mean and std for each classifier
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
    result_filename = f'gokrimp_results/{dataset}_results.csv'
    results_df.to_csv(result_filename)

    print("\nMean and Std of 10 experiments:")
    print(results_df)
    print("\n")
