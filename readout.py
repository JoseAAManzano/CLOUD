# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 14:54:12 2020

@author: josea
"""
# %% Readout of hidden layer for ever
import os
import utils
import torch
import json

import pandas as pd

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

curr_dir = os.getcwd()

try:
    os.stat(os.path.join(curr_dir, 'hidden/'))
except:
    os.mkdir(os.path.join(curr_dir, 'hidden/'))

# %% Hidden representation for each time-point for each word
hidd_cols = [f"hid_{str(i+1)}" for i in range(args.hidden_dims)]
    
for data, category in zip(args.datafiles, args.modelfiles):
    for prob in args.probs:
        end = f"{prob:02}-{100-prob:02}"
        m_name = f"{category}_{end}"

        df = pd.read_csv(f'data/{data}')
        vectorizer = utils.Vectorizer.from_df(df)
        train_df = df[df.split == 'train']
        val_df = df[df.split == 'val']
        test_df = df[df.split == 'test']

        for run in range(args.n_runs):
            t0 = datetime.now()
            print(f"\n{data}: {m_name}_{run}\n")
            cols = ['dataset', 'prob', 'run', 'word', 'label', 'char']

            hidd = defaultdict(list)

            model = torch.load(args.model_save_file +
                               f"{m_name}/{m_name}_{run}_threshold_val_35.pt")
            model.to('cpu')
            model.eval()

            hidd = defaultdict(list)
            for w, l in zip(val_df.data, val_df.label):
                for i, (f_v, t_v) in vectorizer.vectorize_single_char(w):
                    f_v, t_v = f_v.to('cpu'), t_v.to('cpu')
                    hidden = model.init_hidden(1)
                    _, out_rnn, hidden = model(f_v.unsqueeze(0), torch.LongTensor([i+1]), hidden, max_length=i+1)
                    hidd['dataset'].append(category)
                    hidd['prob'].append(end)
                    hidd['run'].append(run)
                    hidd['char'].append(i)
                    hidd['word'].append(w)
                    hidd['label'].append(l)
                    hid = torch.flatten(out_rnn.squeeze(0)[-1].detach()).to('cpu').numpy()
                    for k, v in zip(hidd_cols, hid):
                        hidd[k].append(float(v))

            with open(f"hidden/val_hidden_{m_name}_{run}.json", 'w',
                      encoding='utf-8') as f:
                json.dump(hidd, f)
            
            hidd = defaultdict(list)
            for w, l in zip(test_df.data, test_df.label):
                for i, (f_v, t_v) in vectorizer.vectorize_single_char(w):
                    f_v, t_v = f_v.to('cpu'), t_v.to('cpu')
                    hidden = model.init_hidden(1)
                    _, out_rnn, hidden = model(f_v.unsqueeze(0), torch.LongTensor([i+1]), hidden, max_length=i+1)
                    hidd['dataset'].append(category)
                    hidd['prob'].append(end)
                    hidd['run'].append(run)
                    hidd['char'].append(i)
                    hidd['word'].append(w)
                    hidd['label'].append(l)
                    hid = torch.flatten(out_rnn.squeeze(0)[-1].detach()).to('cpu').numpy()
                    for k, v in zip(hidd_cols, hid):
                        hidd[k].append(float(v))

            with open(f"hidden/test_hidden_{m_name}_{run}.json", 'w',
                      encoding='utf-8') as f:
                json.dump(hidd, f)

            print(f"{(datetime.now() - t0).seconds}s")
