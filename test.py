import os
import torch
import numpy as np
import pandas as pd
from train_function import trainning

def train(data_set):
    # The number of training epochs and patience.
    n_epochs = 100
    patience = 50  # If no improvement in 'patience' epochs, early stop
    batch_size = 64
    kmer_length = 5
    pattern_number = 1024

    total_accuracy = []
    for i in range(10):
        acc = trainning(n_epochs, patience, times=5, batchsize=batch_size,
                        dataset=data_set, kmer_length=kmer_length, patten_number=pattern_number)
        total_accuracy.append(acc)
        print(f'Experiment {i + 1}: {acc}')

    # Calculate mean and std for each classifier
    classifier_names = total_accuracy[0][0].keys()
    results = {name: [] for name in classifier_names}

    for name in classifier_names:
        all_results = [result_set[0][name][0] for result_set in total_accuracy]
        results[name] = [np.mean(all_results), np.std(all_results)]

    results_df = pd.DataFrame(results, index=['mean', 'std']).T
    result_filename = f'result_{data_set}_{batch_size}_{kmer_length}_{pattern_number}.csv'
    results_df.to_csv(result_filename)

    print("\nMean and Std of 10 experiments:")
    print(results_df)

if __name__ == '__main__':
    dataset = ['skating', 'epitope', 'unix', 'question', 'webkb', 'news', 'activity', 'pioneer', 'aslbu', 'robot', 'context', 'auslan2', 'gene', 'reuters']
    for data in dataset:
        train(data)
