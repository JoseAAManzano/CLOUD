# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 12:03:28 2021

@author: josea
"""

# %% Imports
import os
import sys

sys.path.append(os.path.abspath(".."))

import utils
import torch
import json
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
    model_save_file='../models/',
    datafiles=['ESP-ENG.csv', 'ESP-EUS.csv'],
    # Simulation parameters
    modelfiles=['ESEN', 'ESEU'],
    probs=[60, 100],
    n_runs=10,
    # Model hyperparameters
    embedding_dim=32,
    hidden_dims=128,
    n_rnn_layers=1,
    drop_p=0.4,
    # Training hyperparameters
    n_epochs=50,
    learning_rate=0.001,
    batch_size=128,
    # Meta parameters
    acc_threshold=30,
    plotting=False,
    print_freq=10,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    seed=404
)

utils.set_all_seeds(args.seed, args.device)

#%% Read JSON files from each model and run

tmp = defaultdict(list)
splits = ['train', 'val_l1', 'val_l2']
for data, category in zip(args.datafiles, args.modelfiles):
    for prob in args.probs:
        if category == 'ESEU' and prob == 100:
            continue
        
        end = f"{prob:02}-{100-prob:02}"
            
        for run in range(args.n_runs):     
            m_name = f"{category}_{end}"
            
            json_file = f"{args.model_save_file}/{m_name}/{m_name}_{run}.json"
            
            with open(json_file) as f:
                data = json.load(f)
            
            if "100-00" in m_name:
                cat = 'MONO'
            elif "ESEN" in m_name:
                cat = 'ES-EN'
            else:
                cat = 'ES-EU'
                
            for split in splits:
                for i in range(args.n_epochs):
                    tmp['Version'].append(cat)
                    tmp['Run'].append(run)
                    tmp['Split'].append(split)
                    tmp['Epoch'].append(i+1)
                    
                    if split == 'train':
                        tmp['Accuracy'].append(data['train_acc'][i])
                        tmp['Loss'].append(data['train_loss'][i])
                    elif split == 'val_l1':
                        tmp['Accuracy'].append(data['val_acc_l1'][i])
                        tmp['Loss'].append(data['val_loss_l1'][i])
                    else:
                        tmp['Accuracy'].append(data['val_acc_l2'][i])
                        tmp['Loss'].append(data['val_loss_l2'][i])

results = pd.DataFrame(tmp)
results.to_csv('backup_train_accuracy.csv', index=False, encoding='utf-8')
#%% Plot
sns.set(style='whitegrid', context='paper', palette='colorblind', font_scale=2)

results = pd.read_csv('backup_train_accuracy.csv')
res = results[results.Split != 'val_l2']

g = sns.catplot(x='Epoch', y='Accuracy', hue='Version',
                hue_order=['MONO', 'ES-EN', 'ES-EU'], palette=['C1', 'C0', 'C2'],
                col='Split', col_order=['train', 'val_l1'],
                kind='point', data=res, ci=99)
g.axes.flatten()[0].axhline(35, color='k')
g.axes.flatten()[1].axhline(35, color='k')
g.set(xticks=np.arange(-1, 52, 5), title='')
plt.show()

g = sns.catplot(x='Epoch', y='Loss', hue='Version',
                hue_order=['MONO', 'ES-EN', 'ES-EU'], palette=['C1', 'C0', 'C2'],
                col='Split', col_order=['train', 'val_l1'],
                kind='point', data=res, ci=99)
g.set(xticks=np.arange(-1, 52, 5), title='')
plt.show()