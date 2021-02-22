# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 15:14:44 2020

@author: josea
"""
# %% Impports
import sys
import os

sys.path.append(os.path.abspath(".."))

import utils
import math
import torch
import torch.nn as nn
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from argparse import Namespace
from collections import defaultdict

# %% Set-up paramenters
args = Namespace(
    csv='../../processed_data/',
    vectorizer_file="../../processed_data/vectorizer.json",
    model_checkpoint_file='../models/checkpoints/',
    model_save_file='../models/param_search/',
    hidden_dims=[64, 128, 256, 512],
    batch_size=256,
    datafiles=['ESP-ENG.csv', 'ESP-EUS.csv'],
    modelfiles=['ESEN_', 'ESEU_'],
    probs=[1, 20, 40, 50, 60, 80, 99],
    n_runs=5,
    plotting=False,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    seed=404
)

# %% Helper

def cosine_distance(dist1, dist2):
    return dist1.dot(dist2) / (np.linalg.norm(dist1) * np.linalg.norm(dist2))

def sse(dist1, dist2):
    return np.square(dist2 - dist1).sum()

def kl_divergence(dist1, dist2):
    pos = (dist1 != 0.) & (dist2 != 0.)
    return np.sum(dist1[pos] * (np.log2(dist1[pos]) - np.log2(dist2[pos])))

def evaluate(model, loader, criterion):
    running_loss = 0.
    running_acc = 0.

    model.eval()
    for batch_id, batch_dict in enumerate(loader):
        hidden = model.initHidden(args.batch_size, args.device)

        out, hidden = model(batch_dict['X'], hidden)

        loss = loss_func1(*utils.normalize_sizes(out, batch_dict['Y']))

        running_loss += (loss.item() - running_loss) / (batch_id + 1)
        acc_chars = utils.compute_accuracy(
            out, batch_dict['Y'], vectorizer.data_vocab.PAD_idx
        )
        running_acc += (acc_chars - running_acc) / (batch_id + 1)
    return running_loss, running_acc

# %%
res = defaultdict(list)

metrics = {'KL': kl_divergence, 'cosine': cosine_distance,
           'sse': sse}

for data, category in zip(args.datafiles, args.modelfiles):
    for prob in [50, 99]:
        end = f"{prob:02}-{100-prob:02}"

        dataset = utils.TextDataset.load_dataset_and_make_vectorizer(
            args.csv + data,
            p=prob/100, seed=args.seed)
        vectorizer = dataset.get_vectorizer()

        train_words = list(dataset.train_df.data)
        val_words = list(dataset.val_df.data)

        train_trie = utils.Trie()
        train_trie.insert_many(train_words)
        
        val_trie = utils.Trie()
        val_trie.insert_many(val_words)

        ngrams = {}
        for n in range(2, 5):
            ngrams[f"{n}-gram"] = utils.CharNGram(data=train_words, n=n,
                                                  laplace=1)
        for name, m in ngrams.items():
            res['dataset'].append(category[:-1])
            res['prob'].append(end)
            res['hidden'].append(name)
            res['accuracy'].append(m.calculate_accuracy(train_words))
            res['perplexity'].append(m.perplexity(train_words))
            res['val_accuracy'].append(m.calculate_accuracy(val_words))
            res['val_perplexity'].append(m.perplexity(val_words))
            
            train_res = utils.eval_distributions(m, train_trie,
                                                 vectorizer, metrics)
            
            for met, v in train_res.items():
                res["train_" + met].append(v)
                
            val_res = utils.eval_distributions(m, val_trie, vectorizer, metrics)
        
            for met, v in val_res.items():
                res["val_" + met].append(v)
            

        for hidden_units in args.hidden_dims:
            m_name = f"{category}_{hidden_units}_{end}"

            run = 0
            print(f"\n{data}: {m_name}_{run}\n")

            model = torch.load(args.model_save_file +
                               f"{m_name}/{m_name}_{run}" + ".pt")

            loss_func1 = nn.CrossEntropyLoss(
                ignore_index=vectorizer.data_vocab.PAD_idx)

            dataset.set_split('train')
            batch_generator = utils.generate_batches(dataset,
                                                      batch_size=args.batch_size,
                                                      device=args.device)

            res['dataset'].append(category[:-1])
            res['prob'].append(end)
            res['hidden'].append(hidden_units)
            
            loss, acc = evaluate(model, batch_generator, loss_func1)
            
            res['accuracy'].append(acc)
            res['perplexity'].append(math.exp(loss))

            dataset.set_split('val')
            batch_generator = utils.generate_batches(dataset,
                                                      batch_size=args.batch_size,
                                                      device=args.device)
            loss_func1 = nn.CrossEntropyLoss(
                ignore_index=vectorizer.data_vocab.PAD_idx)
            loss, acc = evaluate(model, batch_generator, loss_func1)

            # res['val_loss'].append(loss)
            res['val_accuracy'].append(acc)
            res['val_perplexity'].append(math.exp(loss))

            train_res = utils.eval_distributions(model, train_trie, vectorizer, metrics)
            
            for met, v in train_res.items():
                res["train_" + met].append(v)
            
            val_res = utils.eval_distributions(model, val_trie, vectorizer, metrics)
            
            for met, v in val_res.items():
                res["val_" + met].append(v)

results = pd.DataFrame(res)

results.to_csv('backup_compare_architectures.csv', index=False, encoding='utf-8')

# %%
results = pd.read_csv('backup_compare_architectures.csv')

results_acc = pd.melt(results, id_vars=['dataset', 'prob', 'hidden'],
                      value_vars=['accuracy', 'val_accuracy'],
                      var_name='split', value_name='ACC')
results_acc['split'] = np.where(
    results_acc.split == 'accuracy', 'train', 'val')

g = sns.catplot(x='hidden', y='ACC', hue='split', hue_order=['train', 'val'],
                row='dataset', col='prob', kind='bar',
                data=results_acc, palette='Reds')

results_loss = pd.melt(results, id_vars=['dataset', 'prob', 'hidden'],
                        value_vars=['perplexity', 'val_perplexity'],
                        var_name='split', value_name='Perplexity')
results_loss['split'] = np.where(results_loss.split == 'perplexity', 'train', 'val')

g = sns.catplot(x='hidden', y='Perplexity', hue='split', hue_order=['train', 'val'],
                row='dataset', col='prob', kind='bar',
                data=results_loss, palette='Reds')

for metric in metrics.keys():
    met = pd.melt(results, id_vars=['hidden', 'dataset', 'prob'],
                        value_vars=['train_'+metric, 'val_'+metric],
                        var_name='split', value_name=metric)
    met['split'] = np.where(met.split == 'train_'+metric, 'train', 'val')
    plt.figure()
    g = sns.catplot(x='hidden', y=metric, hue='split', hue_order=['train', 'val'],
                    row='dataset', col='prob', kind='bar',
                    data=met, palette='Reds')