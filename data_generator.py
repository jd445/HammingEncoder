import pandas as pd


from torch.utils.data import Dataset
import sklearn

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
    # conver data_label to int with sklearn
    data_label = sklearn.preprocessing.LabelEncoder().fit_transform(data_label)

    return db, data_label, itemset, max_sequence_length

class SequenceDataset(Dataset):


    def __init__(self, seq_pictures, data_label):
        super(SequenceDataset, self).__init__()
        self.X = seq_pictures
        self.y = data_label

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)
