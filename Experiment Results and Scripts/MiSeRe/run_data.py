import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import subprocess
import timeit
import pandas as pd
from tqdm import tqdm

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
        return patterns
    except FileNotFoundError:
        print("File not found.")
        return None
    except Exception as e:
        print("An error occurred:", e)
        return None
    

       
def generate_the_split(dataset, no_folds = 5):     
            print('Handling dataset:', dataset)

            trace_file = open('datasets/'+dataset+'.dat', 'r')
            label_file = open('datasets/'+dataset+'.lab', 'r')

            traces = []
            label_list = []
            for trace, label in zip(trace_file, label_file):
                traces.append(trace)
                label_list.append(label.replace('\n',''))

            label_set = set(label_list)
            no_labels = len(label_set)
            print('#labels:', no_labels)

            trace_file.close()
            label_file.close()

            #############################################
            ### MiSeRe employs a different file structure
            trace_file = open('datasets/'+dataset+'.dat', 'r')
            label_file = open('datasets/'+dataset+'.lab', 'r')

            traces = []
            label_list = []
            label_dict = {}
            label_no = -3
            for trace, label in zip(trace_file, label_file):
                traces.append(trace)
                label_str = label.replace('\n','')
                if label_str not in label_dict.keys():
                    label_dict[label_str] = str(label_no)
                    label_no -= 1
                label_list.append(label_dict[label_str])

            label_set = set(label_list)
            no_labels = len(label_set)
            print('#labels:', no_labels)

            skf = KFold(n_splits=no_folds, shuffle=True)

            lsev = []
            for fold, (train_index, test_index) in enumerate(skf.split(traces, label_list)):
                trace_file_write = open('datasets/training-test-data/MiSeRe_data/'+dataset+'_training_fold_'+str(fold)+'.text','w')
                for i in train_index:
                    trace_file_write.write(label_list[i]+" -1 "+traces[i])
                trace_file_write.close()

                trace_file_write = open('datasets/training-test-data/MiSeRe_data/'+dataset+'_test_fold_'+str(fold)+'.text','w')
                for i in test_index:
                    trace_file_write.write(label_list[i]+" -1 "+traces[i])
                trace_file_write.close()


def read_dataset(dataset, fold, type):
    # read the dataset and return the data and labels
    X_train = []
    y_train = []
    for line in open('datasets/training-test-data/MiSeRe_data/'+dataset+'_'+type+'_fold_'+str(fold)+'.text'):
        all_list = line.replace('\n','').split(' ')
        # remove -1
        all_list = [x for x in all_list if x != '-1']
        # remove the last element
        all_list = all_list[:-1]
        y_train.append(all_list[0])
        X_train.append(all_list[1:])


    return X_train, y_train



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


#datasets = ['auslan2', 'aslbu','pioneer','skating','question']
datasets = ['context', 'epitope' , 'gene', 'reuters', 'unix', 'robot']
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
                "java -jar MiSeRe.jar -input:datasets/training-test-data/MiSeRe_data/{}_training_fold_{}.text -output:res -run:300s -extract:1024".format(
                    dataset, fold), shell=True)
            print('mining finished')
            #############################################
            #read  MiSeRe features  res
            file_path = './res'  # 文件名是固定的
            patterns = read_patterns(file_path)
            train_X, train_y = read_dataset(dataset, fold, 'training')
            test_x, test_y = read_dataset(dataset, fold, 'test')

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
    result_filename = f'misere_results/{dataset}_results.csv'
    results_df.to_csv(result_filename)

    print("\nMean and Std of 10 experiments:")
    print(results_df)
    print("\n")
