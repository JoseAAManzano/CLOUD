# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 10:54:40 2020

@author: josea
"""
# %% Imports
import os
import utils
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from cloudmodel import CLOUD
from argparse import Namespace
from datetime import datetime
from collections import defaultdict

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
    embedding_dim=32,
    hidden_type='LSTM',
    hidden_dims=128,
    n_rnn_layers=1,
    drop_p=0.4,
    # Training hyperparameters
    n_epochs=50,
    learning_rate=0.001,
    batch_size=128,  # Selected based on train-dev-test sizes
    # Meta parameters
    acc_threshold=30,
    plotting=False,
    print_freq=10,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    seed=404
)

utils.set_all_seeds(args.seed, args.device)

tmp = defaultdict(list)
splits = ['val_l1', 'val_l2', 'test_l1', 'test_l2']
for data, category in zip(args.datafiles, args.modelfiles):
    for prob in args.probs:
        end = f"{prob:02}-{100-prob:02}"
            
        for run in range(args.n_runs):     
            m_name = f"{category}_{end}"
            
            if data == 'ESP-EUS.csv' and prob == args.probs[-1]:
                continue

            if "100-00" in m_name:
                cat = 'MONO'
            elif "ESEN" in m_name:
                cat = 'ES-EN'
            else:
                cat = 'ES-EU'

            model_file = args.model_save_file + f"{args.hidden_type}/{args.hidden_dims}/{m_name}/{m_name}_{run}_threshold_val_35.pt"
            print(f"\nSave file: {data}: {model_file}\n")

            model = torch.load(model_file)
            model.eval()

            for dt in args.datafiles:
                print(dt)
                df = pd.read_csv(args.csv + dt)
                vectorizer = utils.Vectorizer.from_df(df)
                mask_index = vectorizer.data_vocab.PAD_idx
        
                labels = dt.split('.')[0].split('-')
    
                # Set-up dataset, vectorizer, and model
                dataset = utils.TextDataset.make_text_dataset(df, vectorizer,
                                                                p=prob/100,
                                                                labels=labels)
    
                for split in splits:
                    eval_loss, eval_acc = utils.evaluate_model(
                            args, model, split=split, dataset=dataset, mask_index=mask_index, max_length=vectorizer.max_length)
    
                    tmp['category'].append(category)
                    tmp['run'].append(run)
                    tmp['dataset'].append(dt)
                    tmp['Split'].append(split)
                    tmp['Version'].append(cat)
                    tmp['loss'].append(eval_loss)
                    tmp['Accuracy'].append(eval_acc)
            
res = pd.DataFrame(tmp)

mc = res

langs = [x[-2:].upper() + "_" + y.split('.')[0].split('-')[1] for x,y in zip(mc['Split'], mc['dataset'])]

mc['lang'] = langs
mc['Set'] = mc['Split'].map(lambda x: x[:-3].upper())

mc = mc[mc.lang != 'L1_EUS']
mc['Language'] = mc['lang'].map({'L1_ENG': 'Spanish (ES)',
                                 'L2_ENG': 'English (EN)',
                                 'L2_EUS': 'Basque (EU)'})

mc.to_csv('results/model_comparison.csv', index=False, encoding='utf-8')


#%%
mc = pd.read_csv('results/model_comparison.csv')

import seaborn as sns

sns.set(style='whitegrid', context='paper', palette='pastel', font_scale=1.5)

plt.figure(figsize=(6, 9))
ax = sns.catplot(data=mc, x='Set', y='Accuracy', hue='Version',
                 hue_order=['MONO', 'ES-EN', 'ES-EU'],
                 col='Language', kind='bar', ci=99)
plt.show()
