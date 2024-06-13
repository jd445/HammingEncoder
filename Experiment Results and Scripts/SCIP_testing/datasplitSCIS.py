
import os

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
def scis(dataset):
    db, data_label, itemset, max_sequence_length = datainput(dataset)
    # sava the data into len(set(label)) number of txt file
    # each file contains the data with the same label
    label_set = set(data_label)
    for label in label_set:
        f = open(label + '.txt', 'w')
        for i in range(len(data_label)):
            if data_label[i] == label:
                for j in range(len(db[i])):
                    f.write("%s %d\n" % (db[i][j], j+1))
                f.write("\n")
        f.close()
        print(label + ' saved')
    string = ''
    for label in label_set:
        string += label + '.txt,'
    string = string[:-1]
    print(string)
    print('len(set(label)) = ' + str(len(label_set)))
    # if len(label_set) == x, then creat a string 0.2,0.2 *x
    string02 = ''
    for i in range(len(label_set)):
        string02 += '0.02,'
    string02 = string02[:-1]
    string05 = ''
    for i in range(len(label_set)):
        string05 += '0.05,'
    string05 = string05[:-1]

    os.system(f'java -jar -Xmx4G SCIS.jar {len(label_set)} {string} good3 {string02} 1 3 {string05} T_{dataset}.txt 0.05 MC 11 >> Python_{dataset}_SCIS.out')
if __name__ == '__main__':
    for i in range(1, 11):
        dataset = ['auslan2','aslbu','pioneer','context','robot','epitope','skating','question','reuters','gene','unix']
        for i in dataset:
            datasetname = i+'.txt'
            scis(datasetname)
