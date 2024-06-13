import numpy as np
import timeit
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn import svm
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import subprocess
from sklearn.naive_bayes import BernoulliNB
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

### represent a sequence in form of items and sequential patterns (SPs),
### learn sequence vectors using Doc2Vec (PV-DBOW) from items and SPs simultaneously
### use SVM as classifier


def evaluate_classifiers(train_feature, train_label, test_feature, test_label):
    classifiers = {
        "RandomForest": RandomForestClassifier(),
        "SVM": svm.SVC(),
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



### functions ###
# mine SPs from sequences
def mine_SPs(file_seq, minSup, gap, file_seq_items_sp):
    subprocess.run("sp_miner.exe -dataset {} -minsup {} -gap {} -seqsymsp {}".
                   format(file_seq, minSup, gap, file_seq_items_sp))

# load sequences in form of both items and SPs and their labels
def load_seq_items_SPs(file_name):
    labels, sequences = [], []
    with open(file_name) as f:
        for line in f:
            label, content = line.split("\t")
            if content != "\n":
                labels.append(label)
                sequences.append(content.rstrip().split(" "))
    return sequences, labels

# create a sequence id to each sequence
def assign_sequence_id(sequences):
    sequences_with_ids = []
    for idx, val in enumerate(sequences):
        sequence_id = "s_{}".format(idx)
        sequences_with_ids.append(TaggedDocument(val, [sequence_id]))
    return sequences_with_ids



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



dataset = ['auslan2','aslbu','pioneer'	,'context'	,'robot'	,'epitope'	,'skating'	,'question'	,'unix'	,'Gene'	,'reuters']

for data_name in dataset:
    ### variables ###
    total_accuracy = []
    for i in range(10):
        path = "./data/" + data_name
        minSup = 0.05
        gap = 4 # 0: any gap or >0: use gap constraint
        dim = 128
        n_run = 10
        start_date_time = datetime.datetime.now()
        start_time = timeit.default_timer()

        print("### sqn2vec_sim_classify, data: {}, minSup={}, gap={}, dim={} ###".format(data_name, minSup, gap, dim))
        # mine SPs and associate each sequence with a set of items and SPs
        in_seq = path + "/{}.txt".format(data_name)
        out_seq_items_sp = path + "/{}_seq_items_sp_{}_{}.txt".format(data_name, minSup, gap)
        mine_SPs(in_seq, minSup, gap, out_seq_items_sp)
        # load sequences in the form of both items and SPs
        data_path = path + "/{}_seq_items_sp_{}_{}.txt".format(data_name, minSup, gap)
        data_X, data_y = load_seq_items_SPs(data_path)
        # assign a sequence id to each sequence
        data_sen_X = assign_sequence_id(data_X)

        all_acc, all_mic, all_mac = [], [], []


        # learn sequence vectors using Doc2Vec (PV-DBOW) from both items and SPs
        d2v_dbow = Doc2Vec(vector_size=dim, min_count=0, workers=16, dm=0, epochs=50)
        d2v_dbow.build_vocab(data_sen_X)
        d2v_dbow.train(data_sen_X, total_examples=d2v_dbow.corpus_count, epochs=d2v_dbow.epochs)
        data_vec = [d2v_dbow.docvecs[idx] for idx in range(len(data_sen_X))]
        del d2v_dbow  # delete unneeded model memory
        data_vec= np.array(data_vec)
        data_y = np.array(data_y)
        # test performance of classifiers
        x_train_list, x_test_list, y_train_list, y_test_list = k_folder_split(data_vec, data_y, k=5)

        all_res = []
        for j in range(5):
            acc = evaluate_classifiers(x_train_list[j], y_train_list[j], x_test_list[j], y_test_list[j])
            all_res.append(acc)
        total_accuracy.append(all_res)
        # calculate mean and std for each classifier
        # calculate mean and std for each classifier
    classifier_names = total_accuracy[0][0].keys()
    results = {name: [] for name in classifier_names}
    for name in classifier_names:
        acc_10 = []
        for j in range(10):
            acc_5_mean = 0
            for k in range(5):
                acc_5_mean += total_accuracy[j][k][name][0]
            acc_10.append(acc_5_mean/5)
        results[name] = [np.mean(acc_10), np.std(acc_10)]
    results_df = pd.DataFrame(results, index=['mean', 'std']).T
    result_filename = f'results_seqn2sim/{data_name}_results.csv'
    results_df.to_csv(result_filename)

    