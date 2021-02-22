# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 10:53:36 2020

@author: josea
"""
# %%
import sys
import os

sys.path.append(os.path.abspath(".."))

import utils
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from argparse import Namespace

args = Namespace(
    csv='../../processed_data/',
    model_save_file='../models/',
    datafiles=['ESP-ENG.csv', 'ESP-EUS.csv'],
    modelfiles=['ESEN_', 'ESEU_'],
    probs=[1, 20, 40, 50, 60, 80, 99],
    n_runs=5,
    hidden_dim=128,
    learning_rate=0.001,
    device=torch.device('cpu'),
    seed=404
)


# %%
dataset = utils.TextDataset.load_dataset_and_make_vectorizer(
    args.csv + 'ESP-ENG.csv',
    p=50/100, seed=args.seed)
vectorizer = dataset.get_vectorizer()

model = torch.load(args.model_save_file + 'ESEN_50-50/ESEN_50-50_0.pt')
model.eval()

train_words = list(dataset.train_df.data)
test_words = list(dataset.test_df.data)

train_trie = utils.Trie()
train_trie.insert_many(train_words)

test_trie = utils.Trie()
test_trie.insert_many(test_words)

# %%

ngram4 = utils.CharNGram(train_words, n=4, laplace=0.2)

d3 = list(ngram4.get_distribution_from_context('al').values())
d2 = utils.get_distribution_from_context(model, 'al', vectorizer)
d1 = train_trie.get_distribution_from_context('al')
d1_test = test_trie.get_distribution_from_context('al')

letters = list(vectorizer.data_vocab._stoi.keys())[:-3] + [list(vectorizer.data_vocab._stoi.keys())[-2]]

# %%
plt.figure()
plt.bar(letters, d1)
plt.title('Empirical Train')
plt.figure()
plt.bar(letters, d1_test)
plt.title('Empirical Test')
plt.figure()
plt.bar(letters, list(d2))
plt.title('LSTM')
plt.figure()
plt.bar(letters, d3)
plt.title('4-gram')

# %%
def kl_divergence(dist1, dist2):
    pos = (dist1 != 0.) & (dist2 != 0.)
    return np.sum(dist1[pos] * (np.log2(dist1[pos]) - np.log2(dist2[pos])))

metrics = {'KL': kl_divergence}

res1 = utils.eval_distributions(ngram4, test_trie, vectorizer, metrics)
print(res1['KL'])
res2 = utils.eval_distributions(model, test_trie, vectorizer, metrics)
print(res2['KL'])


# %%
import torch.nn.functional as F
device = torch.device('cpu')
utils.set_all_seeds(args.seed, device)


model.to(device)
hidden = model.initHidden(1, device)
model.eval()
f_v, _ = vectorizer.vectorize(list('alarmista'))
f_v = f_v.to(device)
out, hidden = model(f_v.unsqueeze(0), hidden)
dist = torch.flatten(out.detach())[-29:]
# Take only valid continuations (letters + SOS)
ret = torch.empty(27)
ret[:-1] = dist[:-3]
ret[-1] = dist[-2]
ret = F.softmax(ret, dim=0)
print(ret)
plt.figure()
plt.bar(letters, ret)

trie = utils.Trie()
trie.insert_many(train_words)
res2 = utils.eval_distributions(model, trie, vectorizer, metrics)
print(res2['KL'])

d2 = utils.get_distribution_from_context(model, 'alarmista', vectorizer)
d1 = trie.get_distribution_from_context('alarmista')

plt.figure()
plt.bar(letters, d1)
plt.title('Empirical')
plt.figure()
plt.bar(letters, list(d2))
plt.title('LSTM')


d3 = ngram4.get_distribution_from_context('alarmista')

plt.figure()
plt.bar(letters, list(d3.values()))
plt.title('5-gram')



#%%
utils.set_all_seeds(404, device=args.device)
d2 = utils.get_distribution_from_context(model, 'al', vectorizer)
plt.figure()
plt.bar(letters, list(d2))
plt.title('LSTM')