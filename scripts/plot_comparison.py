# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 14:34:52 2021

@author: josea
"""
# %% Imports
import os
import sys

sys.path.append(os.path.abspath(".."))

import utils
import torch
import pandas as pd
import numpy as np

from argparse import Namespace
from collections import defaultdict

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# %% Set-up paramenters
args = Namespace(
    # Path and data information
    csv='../data/',
    model_save_file='architectures/',
    datafiles=['ESP-ENG.csv'],
    # Simulation parameters
    modelfiles=['ESEN'],
    probs=[100],
    n_runs=5,  # How many versions of the models to train
    # Model hyperparameters
    embedding_dim=32,
    hidden_dims=[32, 64, 128, 256, 512],
    n_rnn_layers=1,
    drop_p=0.4,
    # Training hyperparameters
    n_epochs=50,
    learning_rate=0.001,
    batch_size=128,  # Selected based on train-val-test sizes
    # Meta parameters
    plotting=False,
    print_freq=10,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    seed=404
)

utils.set_all_seeds(args.seed, args.device)

# %% Helper

def cosine_distance(dist1, dist2):
    return dist1.dot(dist2) / (np.linalg.norm(dist1) * np.linalg.norm(dist2))


def kl_divergence(dist1, dist2):
    pos = (dist1 != 0.) & (dist2 != 0.)
    return np.sum(dist1[pos] * (np.log2(dist1[pos]) - np.log2(dist2[pos])))

# %%
metrics = {'KL': kl_divergence, 'cosine': cosine_distance}

res_cloud = defaultdict(list)

for data, category in zip(args.datafiles, args.modelfiles):
    for prob in args.probs:
        end = f"{prob:02}-{100-prob:02}"

        df = pd.read_csv(args.csv + data)
        vectorizer = utils.Vectorizer.from_df(df)
        mask_index = vectorizer.data_vocab.PAD_idx
        
        train_words = list(df[(df.label == 'ESP') &
                              (df.split == 'train')].data)
        test_words = list(df[(df.label == 'ESP') &
                             (df.split == 'val') |
                             (df.split == 'test')].data)
        
        train_trie = utils.Trie()
        train_trie.insert_many(train_words)
        
        test_trie = utils.Trie()
        test_trie.insert_many(test_words)
        
        m_name = f"{category}_{end}"
        
        try:
            ngram_results = pd.read_csv(f'ngram_results_{m_name}.csv')
            print(f"N-gram results found for {m_name}! Loading from file.")
        except:
            print(f'Computing n-gram results for {m_name}. This will take a while.')
            res = defaultdict(list)
            for run in range(args.n_runs):
                for n in range(2, 6):
                    print(f'{n}-gram_{run}')
                    ngram = utils.CharNGram(data=train_words, n=n,
                                                      laplace=(run+1)*0.2)
                    
                    train_res = utils.eval_distributions(ngram, train_trie,
                                                         vectorizer, metrics)
                    test_res = utils.eval_distributions(ngram, test_trie,
                                                        vectorizer, metrics)
                    
                    res['model'].append(f'{n}-gram')
                    res['param'].append((run+1)*0.2)
                    res['run'].append(run)
                    
                    for met, v in train_res.items():
                        res[f'train_{met}'].append(v)
                        
                    for met, v in test_res.items():
                        res[f'test_{met}'].append(v)
            del ngram
            ngram_results = pd.DataFrame(res)
            ngram_results.to_csv(f'ngram_results_{m_name}.csv', index=False, encoding='utf-8')
        
        try:
            cloud_res = pd.read_csv(f'cloud_results_{m_name}.csv')
            print(f"CLOUD results found for {m_name}! Loading from file.")
        except:
            print(f'Computing CLOUD results for {m_name}. This will take a while.')
            for hidd in args.hidden_dims:
                for run in range(args.n_runs):
                    print(f"{m_name}_{hidd}_{run}")
                    cloud = torch.load(args.model_save_file + 
                                          f"{m_name}_{hidd}/{m_name}_{hidd}_{run}.pt")
                    cloud.to('cpu')
                    cloud.eval()
                    print('train')
                    train_res = utils.eval_distributions(cloud, train_trie,
                                                          vectorizer, metrics)
                    print('test')
                    test_res = utils.eval_distributions(cloud, test_trie,
                                                        vectorizer, metrics)
                    
                    res_cloud['model'].append(f'CLOUD_{hidd}')
                    res_cloud['param'].append(hidd)
                    res_cloud['run'].append(run)
                    
                    for met, v in train_res.items():
                        res_cloud[f'train_{met}'].append(v)
                        
                    for met, v in test_res.items():
                        res_cloud[f'test_{met}'].append(v)
                        
            cloud_res = pd.DataFrame(res_cloud)
            cloud_res.to_csv(f'cloud_results_{m_name}.csv', index=False,
                             encoding='utf-8')            

results = pd.concat([ngram_results, cloud_res], axis=0)

results.to_csv('backup_compare_architectures.csv', index=False, encoding='utf-8')

# %%
sns.set(style='whitegrid', context='paper', palette='colorblind', font_scale=1.5)

results = pd.read_csv('backup_compare_architectures.csv')

mp = {'2-gram':'2g',
      '3-gram':'3g',
      '4-gram':'4g',
      '5-gram':'5g',
      'CLOUD_32':'C32',
      'CLOUD_64':'C64',
      'CLOUD_128':'C128',
      'CLOUD_256':'C256',
      'CLOUD_512':'C512'}

results['model_abc'] = results.model.map(mp)

for metric in metrics.keys():
    met = pd.melt(results, id_vars=['model_abc', 'param', 'run'],
                        value_vars=['train_'+metric, 'test_'+metric],
                        var_name='split', value_name=metric)
    met['split'] = np.where(met.split == 'train_'+metric, 'train', 'test')
    plt.figure(figsize=(12, 10))
    g = sns.catplot(x='model_abc', y=metric, col='split', col_order=['train', 'test'],
                    kind='bar', ci=None, data=met, palette='viridis')
    g.set(title='', xlabel='Model')
    plt.show()