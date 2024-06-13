import pandas as pd

import gc
import os
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


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
    # itemset = set([item for sublist in db for item in sublist])
    # itemset = list(itemset)
    # int_itemset = [str(x) for x in itemset]
    # int_itemset.sort()
    # itemset = [str(x) for x in int_itemset]
    # print(itemset)
    # save itemset as text for decoulping
    # with open('itemset.txt', 'w') as f:
    #     for item in itemset:
    #         f.write("%s " % item)
    # f.close()
    # print('itemset saved')

    return db, data_label, itemset, max_sequence_length
# read all name of txt file in the folder


def read_file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        # remove suffix
        files = [i.split('.')[0] for i in files]

        return files


dataset = read_file_name('data')
print(dataset)
for i in dataset:
    print(i)

    db, data_label, itemset, max_sequence_length = datainput(
        'data/' + i + '.txt')
    # convert datalabel into int
    dict_label = {}
    dict_label = dict(zip(set(data_label), range(len(set(data_label)))))
    data_label = [dict_label[x] for x in data_label]
    # convert db into int
    dict_db = {}
    dict_db = dict(zip(set([item for sublist in db for item in sublist]), range(
        len(set([item for sublist in db for item in sublist])))))
    db = [[dict_db[x] for x in sublist] for sublist in db]
    # print('data_label', data_label)
    # save data_label , db as txt
    with open('split/' + i + '.lab', 'w') as f:
        for item in data_label:
            # every line one label
            f.write("%s \n" % item)
    f.close()
    with open('split/' + i + '.dat', 'w') as f:
        for item in db:
            item = [str(x) for x in item]
            # every line one sequence
            f.write("%s \n" % ' '.join(item))
    f.close()
