# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 11:02:36 2020

@author: josea
"""
import sys
import os

sys.path.append(os.path.abspath(".."))

import utils
import torch
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import cosine, euclidean
from argparse import Namespace
from collections import defaultdict

args = Namespace(
    csv='../../processed_data/',
    model_save_file='../models/',
    datafiles=['ESP-ENG.csv', 'ESP-EUS.csv'],
    modelfiles=['ESEN_', 'ESEU_'],
    probs=[50, 100],
    n_runs=5,
    hidden_dim=128,
    learning_rate=0.001,
    device=torch.device('cpu'),
    seed=404
)

utils.set_all_seeds(args.seed, args.device)
# %% Empirical evaluation

def cosine_distance(dist1, dist2):
    return cosine(dist1, dist2)

def sse(dist1, dist2):
    return euclidean(dist1, dist2)

def kl_divergence(dist1, dist2):
    pos = (dist1 != 0.) & (dist2 != 0.)
    return np.sum(dist1[pos] * (np.log2(dist1[pos]) - np.log2(dist2[pos])))


# %%
def empirical_evaluation(args, metrics):
    res = defaultdict(list)
    for data, category in zip(args.datafiles, args.modelfiles):
        for prob in args.probs:
            end = f"{prob:02}-{100-prob:02}"
            m_name = f"{category}{end}"

            dataset = utils.TextDataset.load_dataset_and_make_vectorizer(
                args.csv + data,
                p=prob/100, seed=args.seed)
            vectorizer = dataset.get_vectorizer()
    
            train_words = list(dataset.train_df.data)
            test_words = list(dataset.test_df.data)
    
            train_trie = utils.Trie()
            train_trie.insert_many(train_words)
            
            test_trie = utils.Trie()
            test_trie.insert_many(test_words)

            for run in range(args.n_runs):
                print(f"\n{data}: {m_name}_{run}\n")

                ngrams = {}
                for n in range(2, 5):
                    ngrams[f"{n}-gram"] = utils.CharNGram(data=train_words, n=n,
                                                          laplace=(run+1)*0.2)

                lstm_model = torch.load(args.model_save_file +
                                        f"{m_name}/{m_name}_{run}.pt")
                lstm_model.to(args.device)
                lstm_model.eval()

                ngrams['LSTM'] = lstm_model
                
                for model, m in ngrams.items():
                    res['model'].append(model)
                    res['dataset'].append(category[:-1])
                    res['prob'].append(prob)
                    res['run'].append(run)
                    
                    train_res = utils.eval_distributions(m, train_trie,
                                                         vectorizer, metrics)
                    for met, v in train_res.items():
                        res["train_" + met].append(v)
                    
                    test_res = utils.eval_distributions(m, test_trie,
                                                        vectorizer, metrics)
                    for met, v in test_res.items():
                        res["test_" + met].append(v)
    return res

# %% Driver code
metrics = {'KL': kl_divergence, 'cosine': cosine_distance,
           'sse': sse}

results = empirical_evaluation(args, metrics)

results = pd.DataFrame(results)

results.to_csv('backup_empirical_eval.csv', index=False, encoding='utf-8')

#%%
results = pd.read_csv('backup_empirical_eval.csv')

for metric, name in zip(metrics.keys(), ['KL', 'cosine', 'euclidean']):
    met = pd.melt(results, id_vars=['model', 'dataset', 'prob'],
                       value_vars=['train_'+metric, 'test_'+metric],
                       var_name='split', value_name=metric)
    met['split'] = np.where(met.split == 'train_'+metric, 'train', 'test')
    plt.figure()
    g = sns.catplot(x='model', y=metric, hue='split', hue_order=['train', 'test'],
                    row='dataset', col='prob', kind='bar',
                    data=met, palette='Blues')
    for ax in g.axes.flatten()[::2]:
        ax.set_ylabel(name, fontsize=15)
