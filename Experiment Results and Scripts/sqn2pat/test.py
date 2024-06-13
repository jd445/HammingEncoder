from multiprocessing import Pool
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import tree
# from sequential.pat2feat import Pat2Feat
# Example to show how to run Dichotomic Pattern Mining 
# on sequences with positive and negative outcomes
from sequential.seq2pat import Seq2Pat

from sequential.dpm import dichotomic_pattern_mining, DichotomicAggregation


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


def datainput(filepath):
    # db: list form of a sequence
    # strdb: string form of a sequence
    # data_label: list of the label of the sequence
    # itemset: the itemset of the sequence
    # max_sequence_length: the maximum length of the sequence in a dataset.
    max_sequence_length = 0
    file = open(filepath)
    db = []
    data_label = []
    itemset = []
    for i in file:
        temp = i.replace("\n", "").split("\t")
        seq_db = temp[1].split(" ")
        max_sequence_length = max(max_sequence_length, len(seq_db))
        db.append(seq_db)
        data_label.append(str(temp[0]))
    # unique itemset
    itemset = set([item for sublist in db for item in sublist])
    itemset = list(itemset)
    int_itemset = [str(x) for x in itemset]
    int_itemset.sort()
    itemset = [str(x) for x in int_itemset]
    # print(itemset)
    # save itemset as text for decoulping
    # with open('itemset.txt', 'w') as f:
    #     for item in itemset:
    #         f.write("%s " % item)
    # f.close()
    # conver data_label to int with sklearn
    data_label = sklearn.preprocessing.LabelEncoder().fit_transform(data_label)

    return db, data_label, itemset, max_sequence_length

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


def ovo_mining_discriminative_pattern(seqs_by_class):
    seq2pat_by_class = {}
    for label, seqs in seqs_by_class.items():
        seq2pat_by_class[label] = Seq2Pat(sequences=seqs)
    
    dpm_all = []
    for label1 in seqs_by_class.keys():
        for label2 in seqs_by_class.keys():
            if label1 >= label2:
                continue
            aggregation_to_patterns = dichotomic_pattern_mining(seq2pat_by_class[label1], seq2pat_by_class[label2], min_frequency_pos=int(0.2 * len(seqs_by_class[label1])), min_frequency_neg=int(0.2 * len(seqs_by_class[label2])))
            dpm_all.append(aggregation_to_patterns[DichotomicAggregation.union])
    
    # 去重过程
    unique_patterns = set(tuple(pattern) for patterns in dpm_all for pattern in patterns)
    # 将元组列表转回到列表的列表
    unique_patterns_list = [list(pattern) for pattern in unique_patterns]
    # max length of the pattern =5
    unique_patterns_list = [pattern for pattern in unique_patterns_list if len(pattern) <= 5]
    return unique_patterns_list

def drop_frequency(result):
    """
    Drop the frequency appended to each mined pattern.

    Parameters
    ----------
    result: List[list]
        The mined patterns with each one having the count appended to the end

    Returns
    -------
    The list of mined patterns without appended frequency

    """
    return list(map(lambda x: x[:-1], result))

class Pat2Feat:
    """
    **Pat2Feat: Pat2Feat: Pattern-to-Feature Generation**

    """

    def __init__(self):

        # Initialize the implementer of one-hot encodings generation
        self._imp = None

    def get_features(self, sequences, patterns,max_span = 10,drop_pattern_frequency= True):

        if drop_pattern_frequency:
            patterns = drop_frequency(patterns)
        return self.transform_max_parallel(max_span, sequences, patterns, 16)
    
    @staticmethod
    def is_subsequence(pattern, trace):
        it = iter(trace)
        return all(item in it for item in pattern)

    @staticmethod
    def process_chunk(args):
        gap, sequences, patterns = args
        feature = np.zeros((len(sequences), len(patterns)))
        for i, seq in enumerate(sequences):
            for j, pattern in enumerate(patterns):
                for start in range(len(seq) - len(pattern) + 1):
                    end = start + len(pattern) + gap
                    if end > len(seq):
                        continue
                    if Pat2Feat.is_subsequence(pattern, seq[start:end]):
                        feature[i, j] = 1
                        break
        return feature

    def transform_max_parallel(self, gap, sequences, patterns, num_processes):
        # Split sequences into chunks for each process
        chunk_size = len(sequences) // num_processes
        chunks = [sequences[i:i + chunk_size] for i in range(0, len(sequences), chunk_size)]
        
        # Prepare arguments for each chunk
        args = [(gap, chunk, patterns) for chunk in chunks]
        
        # Create a pool of processes
        with Pool(num_processes) as pool:
            results = pool.map(self.process_chunk, args)
        
        # Combine the results from each process
        feature_matrix = np.vstack(results)
        return feature_matrix

datasets = ['aslbu', 'auslan2', 'pioneer'	,'context'	,'robot'	,'epitope'	,'skating'	,'question'	,'unix'	,'gene'	,'reuters']
    # load 
for dataset in datasets:
    total_accuracy = []
    for time in range(10):
        db, data_label, itemset, max_sequence_length = datainput('dataset/' + dataset + '.txt')
        # split data
        x_train_list, x_test_list, y_train_list, y_test_list = k_folder_split(db, data_label, k=5)
        # evaluate
        accuracies = []
        for i in range(5):
            train_feature = x_train_list[i]
            train_label = y_train_list[i]
            test_feature = x_test_list[i]
            test_label = y_test_list[i]
            # 1. 将每个类别的数据打包
            seqs_by_class = {}
            for seq, label in zip(train_feature, train_label):
                if label not in seqs_by_class:
                    seqs_by_class[label] = []
                seqs_by_class[label].append(seq)
            # ovo mining all sequential pattern
            dis_patterns = ovo_mining_discriminative_pattern(seqs_by_class)
            pat2feat = Pat2Feat()
            print(len(dis_patterns))
            train_feature = pat2feat.get_features(train_feature, dis_patterns, drop_pattern_frequency=True)
            test_feature = pat2feat.get_features(test_feature, dis_patterns, drop_pattern_frequency=True)
            # evaluate_classifiers
            acc = evaluate_classifiers(train_feature, train_label, test_feature, test_label)
            accuracies.append(acc)
                # average the result
        total_accuracy.append(accuracies)
    # calculate mean and std for each classifier
    classifier_names = total_accuracy[0][0].keys()
    results = {name: [] for name in classifier_names}

    for name in classifier_names:

        all_mean = []
        for i in range(10):
            mean = 0
            # all_mean.append(total_accuracy[i][0][name][0])
            for j in range(5):
                mean += total_accuracy[i][j][name][0]
            all_mean.append(mean/5)
        results[name] = [np.mean(all_mean), np.std(all_mean)]

    results_df = pd.DataFrame(results, index=['mean', 'std']).T
    result_filename = f'results_sqn2pat/{dataset}_results.csv'
    results_df.to_csv(result_filename)

    print("\nMean and Std of 10 experiments:")
    print(results_df)
    print("\n")

