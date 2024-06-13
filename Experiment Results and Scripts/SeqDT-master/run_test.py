# randomly split the dat into 5 fold and save it
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

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
    with open('itemset.txt', 'w') as f:
        for item in itemset:
            f.write("%s " % item)
    f.close()
    print('itemset saved')

    return db, data_label, itemset, max_sequence_length

datasets = ['Human'	,'cov', 'Anticancer_peptides']
for dataset in datasets:
    db, data_label, itemset, max_sequence_length = datainput(f'SeqDT-master/data/{dataset}.txt')
    for i in range(10):
        # 5 fold
        fold = 5
        # split the data into 5 fold
        kf = KFold(n_splits=fold, shuffle=True)
        kf.get_n_splits(db)
        fold_count = 0
        for train_index, test_index in kf.split(db):
            fold_count += 1
            print("Fold", fold_count)
            train_data = []
            train_label = []
            test_data = []
            test_label = []
            for i in train_index:
                train_data.append(db[i])
                train_label.append(data_label[i])
            for i in test_index:
                test_data.append(db[i])
                test_label.append(data_label[i])
            # save the data
            with open('train_data_fold' + str(fold_count) + '.txt', 'w') as f:
                for i in range(len(train_data)):
                    f.write("%s\t" % train_label[i])
                    for j in range(len(train_data[i])):
                        f.write("%s " % train_data[i][j])
                    f.write("\n")
            f.close()
            with open('test_data_fold' + str(fold_count) + '.txt', 'w') as f:
                for i in range(len(test_data)):
                    f.write("%s\t" % test_label[i])
                    for j in range(len(test_data[i])):
                        f.write("%s " % test_data[i][j])
                    f.write("\n")
            f.close()
            

            print('train data saved')
            print('test data saved')

        import os
        os.system(f' cd /mnt/c/Users/14153/OneDrive\ -\ mail.dlut.edu.cn/HammingEncoder/复现 ; /usr/bin/env /usr/lib/jvm/java-19-openjdk-amd64/bin/java -XX:+ShowCodeDetailsInExceptionMessages -cp /home/junjie/.vscode-server/data/User/workspaceStorage/7d63f3e5a09716a0ef249eb099bd64c3/redhat.java/jdt_ws/复现_eac3ec11/bin SeqDTCode.Main  {dataset}')

