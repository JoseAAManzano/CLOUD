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
    csv='data/',
    model_save_file='models/',
    datafiles=['ESP-ENG.csv', 'ESP-EUS.csv'],
    # Simulation parameters
    modelfiles=['ESEN', 'ESEU'],
    probs=[60, 100],
    n_runs=10,  # How many versions of the models to train
    # Model hyperparameters
    embedding_dim=16,
    hidden_dims=128,
    n_rnn_layers=1,
    drop_p=0.0,
    # Training hyperparameters
    n_epochs=100,
    learning_rate=2e-3,
    batch_size=82,  # Selected based on train-val-test sizes
    # Meta parameters
    acc_threshold=65,
    plotting=False,
    print_freq=10,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    seed=404,
    # LDT
    run_ldt=True,
    ldt_path='data/LDT.csv'
)

utils.set_all_seeds(args.seed, args.device)

#%% Read JSON files from each model and run

tmp = defaultdict(list)
splits = ['LDT_train_score', 'LDT_val_score']
for data, category in zip(args.datafiles, args.modelfiles):
    for prob in args.probs:
        if category == 'ESEU' and prob == 100:
            continue
        
        end = f"{prob:02}-{100-prob:02}"
            
        for run in range(args.n_runs):     
            m_name = f"{category}_{end}"
            
            json_file = f"../{args.model_save_file}/{m_name}/{m_name}_{run}.json"
            
            with open(json_file) as f:
                data = json.load(f)
            
            if "100-00" in m_name:
                cat = 'MONO'
            elif "ESEN" in m_name:
                cat = 'SP-EN'
            else:
                cat = 'SP-BQ'
                
            for split in splits:
                for i in range(args.n_epochs):
                    tmp['Version'].append(cat)
                    tmp['Run'].append(run)
                    tmp['Split'].append(split)
                    tmp['Epoch'].append(i+1)
                    
                    if split == 'LDT_train_score':
                        tmp['Accuracy'].append(data['LDT_train_score'][i])
                        tmp['Loss'].append(data['train_loss'][i])
                    elif split == 'LDT_val_score':
                        tmp['Accuracy'].append(data['LDT_val_score'][i])
                        tmp['Loss'].append(data['val_loss_l1'][i])

results = pd.DataFrame(tmp)
results.to_csv('backup_train_accuracy.csv', index=False, encoding='utf-8')
#%% Plot
sns.set(style='white', context='paper', font_scale=2)

res = pd.read_csv('backup_train_accuracy.csv')

g = sns.catplot(x='Epoch', y='Accuracy', hue='Version',
                hue_order=['MONO', 'SP-EN', 'SP-BQ'], palette=["#666666", "#27AAE1", "#074C7A"],
                col='Split', col_order=['LDT_train_score', 'LDT_val_score'],
                kind='point', data=res, ci=95)
g.axes.flatten()[0].axhline(50, color='k', linestyle='--', lw=5)
g.axes.flatten()[0].set(ylabel='Accuracy (%)')
g.axes.flatten()[1].axhline(50, color='k', linestyle='--', lw=5)
g.axes.flatten()[0].axhline(85, color='r', lw=5)
g.set(xticks=np.arange(-1, 101, 10), title='')
plt.show()

g = sns.catplot(x='Epoch', y='Loss', hue='Version',
                hue_order=['MONO', 'SP-EN', 'SP-BQ'], palette=["#666666", "#27AAE1", "#074C7A"],
                col='Split', col_order=['LDT_train_score', 'LDT_val_score'],
                kind='point', data=res, ci=95)
g.axes.flatten()[0].axhline(1.8, color='r', lw=5)
g.set(xticks=np.arange(-1, 101, 10), title='')
plt.show()