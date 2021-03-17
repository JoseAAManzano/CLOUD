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
    batch_size=128,  # Selected based on train-val-test sizes
    # Meta parameters
    acc_threshold=30,
    plotting=False,
    print_freq=10,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    seed=404
)

utils.set_all_seeds(args.seed, args.device)

for data, category in zip(args.datafiles, args.modelfiles):
    for prob in args.probs:
        # No need to run twice the monolingual versions
        if data == 'ESP-EUS.csv' and prob == args.probs[-1]:
            continue

        end = f"{prob:02}-{100-prob:02}"

        df = pd.read_csv(args.csv + data)
        vectorizer = utils.Vectorizer.from_df(df)
        mask_index = vectorizer.data_vocab.PAD_idx

        for run in range(args.n_runs):
            threshold = args.acc_threshold
            threshold_val = args.acc_threshold

            m_name = f"{category}_{end}"

            t0 = datetime.now()

            save_file = args.model_save_file + f"{m_name}"
            print(f"\nSave file: {data}: {save_file}/{m_name}_{run}\n")

            train_state = utils.make_train_state(
                f"{save_file}/{m_name}_{run}.pt")

            model = CLOUD(
                char_vocab_size=len(vectorizer.data_vocab),
                n_embedd=args.embedding_dim,
                n_hidden=args.hidden_dims,
                n_layers=args.n_rnn_layers,
                hidden_type=args.hidden_type,
                pad_idx=mask_index
            ).to(args.device)

            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.learning_rate)

            l1 = 0
            labels = data[:-4].split('-')

            # Set-up dataset, vectorizer, and model
            dataset = utils.TextDataset.make_text_dataset(df, vectorizer,
                                                          p=prob/100,
                                                          labels=labels)
            # Training loop
            for it in range(args.n_epochs):
                if (it+1) % args.print_freq == 0:
                    print(f"Epoch: {it+1:03d} |",
                          f"Avg. train acc: {np.mean(train_state['train_acc'][-args.print_freq:]):.2f} |",
                          f"Avg. val acc L1: {np.mean(train_state['val_acc_l1'][-args.print_freq:]):.2f} |",
                          f"Avg. val acc L2: {np.mean(train_state['val_acc_l2'][-args.print_freq:]):.2f}")

                train_state['epoch_idx'] = it + 1

                dataset.set_split('train')
                batch_generator = utils.generate_batches(dataset,
                                                         batch_size=args.batch_size,
                                                         device=args.device)
                running_loss = 0.
                running_acc = 0.

                model.train()
                for batch_id, batch_dict in enumerate(batch_generator):
                    optimizer.zero_grad()

                    hidden = model.init_hidden(args.batch_size, args.device)

                    out, _, hidden = model(batch_dict['X'], batch_dict['vector_length'],
                                           hidden, drop_rate=args.drop_p, max_length=vectorizer.max_length)

                    loss = utils.compute_loss(
                        out, batch_dict['Y'], mask_index)

                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    optimizer.step()

                    # Update train_state arguments
                    running_loss += (loss.item() -
                                     running_loss) / (batch_id + 1)
                    acc_chars = utils.compute_accuracy(
                        out, batch_dict['Y'], mask_index)

                    running_acc += (acc_chars - running_acc) / (batch_id + 1)

                train_state['train_loss'].append(running_loss)
                train_state['train_acc'].append(running_acc)

                # EVAL
                eval_loss_l1, eval_acc_l1 = utils.evaluate_model(
                    args, model, split='val_l1', dataset=dataset, mask_index=mask_index, max_length=vectorizer.max_length)
                eval_loss_l2, eval_acc_l2 = utils.evaluate_model(
                    args, model, split='val_l2', dataset=dataset, mask_index=mask_index, max_length=vectorizer.max_length)

                train_state['val_loss_l1'].append(eval_loss_l1)
                train_state['val_acc_l1'].append(eval_acc_l1)

                train_state['val_loss_l2'].append(eval_loss_l2)
                train_state['val_acc_l2'].append(eval_acc_l2)

                # Save best model
                if train_state['epoch_idx'] == 1:
                    try:
                        os.stat(save_file)
                    except:
                        os.mkdir(save_file)

                joint_loss = np.mean(
                    [train_state['val_loss_l1'][-1], train_state['val_loss_l2'][-1]])
                joint_acc = np.mean(
                    [train_state['val_acc_l1'][-1], train_state['val_acc_l2'][-1]])
                if joint_loss < train_state['best_joint_loss']:
                    torch.save(
                        model, f"{save_file}/{m_name}_{run}_best_loss.pt")
                    train_state['best_joint_loss'] = joint_loss
                    train_state['best_epoch_idx_loss'] = train_state['epoch_idx']
                if joint_acc > train_state['best_joint_acc']:
                    torch.save(
                        model, f"{save_file}/{m_name}_{run}_best_acc.pt")
                    train_state['best_joint_acc'] = joint_acc
                    train_state['best_epoch_idx_acc'] = train_state['epoch_idx']

                if train_state['train_acc'][-1] >= threshold:
                    print(
                        f"Train threshold {threshold} reached at epoch {it+1}: {train_state['train_acc'][-1]:.2f}")
                    torch.save(
                        model, f"{save_file}/{m_name}_{run}_threshold_train_{threshold}.pt")
                    threshold += 1
                if train_state['val_acc_l1'][-1] >= threshold_val:
                    print(
                        f"Validation threshold {threshold_val} reached at epoch {it+1}: {train_state['val_acc_l1'][-1]:.2f}")
                    torch.save(
                        model, f"{save_file}/{m_name}_{run}_threshold_val_{threshold_val}.pt")
                    threshold_val += 1

            # TEST
            test_loss_l1, test_acc_l1 = utils.evaluate_model(
                args, model, split='test_l1', dataset=dataset, mask_index=mask_index, max_length=vectorizer.max_length)
            test_loss_l2, test_acc_l2 = utils.evaluate_model(
                args, model, split='test_l2', dataset=dataset, mask_index=mask_index, max_length=vectorizer.max_length)

            train_state['test_loss_l1'] = test_loss_l1
            train_state['test_acc_l1'] = test_acc_l1

            train_state['test_loss_l2'] = test_loss_l2
            train_state['test_acc_l2'] = test_acc_l2

            train_state['run_time'] = f"{(datetime.now() - t0).seconds}s"

            utils.print_state_dict(train_state)
            utils.save_train_state(
                train_state, f"{save_file}/{m_name}_{run}.json")

            torch.save(model, train_state['model_save_file'])

            if args.plotting:
                plt.plot(train_state['train_loss'], label='train_loss')
                plt.plot(train_state['val_loss_l1'], label='val_loss_l1')
                plt.plot(train_state['val_loss_l2'], label='val_loss_l2')
                plt.legend()
                plt.show()

                plt.plot(train_state['train_acc'], label='train_acc')
                plt.plot(train_state['val_acc_l1'], label='val_acc_l1')
                plt.plot(train_state['val_acc_l2'], label='val_acc_l2')
                plt.legend()
                plt.show()
