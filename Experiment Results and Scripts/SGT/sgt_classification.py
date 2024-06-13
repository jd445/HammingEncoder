import pandas as pd
from sklearn import naive_bayes
from sklearn import svm
from sklearn import neighbors
from sklearn import tree
from sklearn.model_selection import KFold
import gc
from sgt import SGT
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
import sklearn
from sklearn import metrics
from tqdm.auto import tqdm

# dataset =  ['pioneer', 'aslbu', 'robot', 'context', 'auslan2', 'gene','cov','Human','Anticancer_peptides']
dataset = ['epitope','skating']
# dataset = ['skating']
result = pd.DataFrame()
clumns = ['RF', 'SVM', 'NB', 'KNN', 'DT']


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

for curr_dataset in dataset:
    try:
        print(curr_dataset)

        corpus = pd.read_csv('dataset/{}.txt'.format(curr_dataset) , sep='\t', header=None)
        
        data_label = corpus[0].values
        corpus.insert(0, 'Entry', range(0, 0 + len(corpus)))
        
        # delete the second column
        corpus = corpus.drop([0], axis=1)
        
        corpus.columns = ['id', 'sequence']
        # convert sequence to list base on ,
        corpus['sequence'] = corpus['sequence'].map(lambda x: x.split(' '))

        sgt_ = SGT(kappa=1, 
                lengthsensitive=False, 
                mode='multiprocessing')
        sgtembedding_df = sgt_.fit_transform(corpus)
        print('sgt finish')
        X = sgtembedding_df.set_index('id')
        total_accuracy = []

        for i in tqdm(range(10)):
            all_res = []
            # 5 folder corss validation with classificaiton. data: X, label: data_label
            kf = KFold(n_splits=5, shuffle=True)
            kf.get_n_splits(X)
            
            for train_index, test_index in tqdm(kf.split(X)):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = np.array(data_label)[train_index], np.array(data_label)[test_index]
                # test performance of classifiers
                acc = evaluate_classifiers(X_train, y_train, X_test, y_test)
                
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
        result_filename = f'results_sgt/{curr_dataset}_results.csv'
        results_df.to_csv(result_filename)

        print("\nMean and Std of 10 experiments:")
        print(results_df)
        print("\n")
    except Exception as e:
        print(e)
        continue