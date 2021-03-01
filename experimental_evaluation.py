# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 09:49:10 2020

@author: josea
"""
# %% Imports
import utils
import torch
import torch.nn as nn
import pandas as pd

import seaborn as sns
import torch.nn.functional as F

from argparse import Namespace
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

# %% Model extension to predict category


class Ensemble(nn.Module):
    def __init__(self, cloud, n_input, n_output):
        super(Ensemble, self).__init__()
        self.cloud = cloud
        self.out = nn.Linear(n_input, n_output)

    def forward(self, x, X_lengths, hidden, batch_size=1, drop_rate=0.0):
        x1, out_rnn, hid = self.cloud(x, X_lengths, hidden, drop_rate)
        inp2 = torch.flatten(out_rnn.squeeze(0)[-1])
        x2 = self.out(F.dropout(inp2, drop_rate))
        return x1, x2


# %% Use test words to mimic learning during the blocks
utils.set_all_seeds(args.seed, args.device)    
    
eval_words = pd.read_csv(args.csv + 'EXP_WORDS.csv')

vectorizer = utils.Vectorizer.from_df(eval_words)

mask_index = vectorizer.data_vocab.PAD_idx

reps_per_block = 2
familiarization = 0

def score_word(prod, target):
    score = 0
    if prod == target:
        score = 100
    else:
        for i, l in enumerate(prod):
            if i > len(target)-1:
                score -= 20
            elif l == target[i]:
                score += 20
    return score if 0 <= score <= 100 else 0

res = defaultdict(list)
for data, category in zip(args.datafiles, args.modelfiles):
    for prob in args.probs:
        if category == "ESEU" and prob == args.probs[-1]:
            continue
        end = f"{prob:02}-{100-prob:02}"
        m_name = f"{category}_{end}"

        if prob == args.probs[0]:
            if category == 'ESEN':
                grp = 'ES-EN'
            else:
                grp = 'ES-EU'
        else:
            grp = 'MONO'

        for run in range(args.n_runs):
            print(f"\n{data}: {m_name}_{run}\n")
            
            exp_words = eval_words
            exp_words = exp_words.sample(frac=1., replace=False)
            exp_words['cat'] = range(48)
            
            model = torch.load(args.model_save_file +
                               f"{args.hidden_type}/{args.hidden_dims}/{m_name}/{m_name}_{run}_threshold_val_34.pt")

            ensemble = Ensemble(model, args.hidden_dims, 48)  
            ensemble.to(args.device)
            ensemble.train()
            
            optimizer = torch.optim.Adam(
                    ensemble.parameters(),
                    lr=args.learning_rate)
            
            if familiarization:
                exp_words = exp_words.sample(frac=1., replace=False)
                for word, cat, lab in zip(exp_words.data, exp_words.cat, exp_words.label):
                    optimizer.zero_grad()
                    loss = 0
                    for i, (f_v, t_v) in vectorizer.vectorize_single_char(word):
                            optimizer.zero_grad()
                            f_v, t_v = f_v.to(args.device), t_v.to(args.device)
                            hidden = ensemble.cloud.init_hidden(1, args.device)
                            
                            out_letters, out_cat = ensemble(f_v.unsqueeze(0), 
                                                            torch.LongTensor([i+1]),
                                                            hidden, 
                                                            args.drop_p)
                            
                            loss += F.cross_entropy(out_letters[-1].unsqueeze(0), t_v,
                                                    ignore_index=mask_index,
                                                    reduction='sum')
                    target = torch.LongTensor([cat]).to(args.device)
                    loss += F.cross_entropy(out_cat.unsqueeze(0), target,
                                           reduction='sum')
                    loss.backward()
                    optimizer.step()
            
            for it in range(5):
                for tr in range(reps_per_block):
                    exp_words = exp_words.sample(frac=1., replace=False)
                    
                    # Recognition task
                    for word, cat, lab in zip(exp_words.data, exp_words.cat, exp_words.label):
                        # preds = []
                        # for i, (f_v, t_v) in vectorizer.vectorize_single_char(word):
                        #     optimizer_cat.zero_grad()

                        #     f_v, t_v = f_v.to(args.device), t_v.to(args.device)
                        #     hidden = ensemble.cloud.init_hidden(1, args.device)
                            
                        #     _, out_cat = ensemble(f_v.unsqueeze(0), 
                        #                                     torch.LongTensor([i+1]),
                        #                                     hidden, 
                        #                                     args.drop_p)
                            
                        #     target = torch.LongTensor([cat]).to(args.device)
                        
                        #     loss = F.cross_entropy(out_cat.unsqueeze(0), target, reduction='sum')
                            

                            
                        #     _, pr = torch.max(F.log_softmax(out_cat, dim=0), dim=0)
                            
                        #     preds.append(pr.item())
                        
                        #     loss.backward()
                        #     optimizer_cat.step()
                        
                        # acc_cat = (max(set(preds), key=preds.count) == cat) * 100
                            
                        optimizer.zero_grad()
                        f_v, t_v, lengths = vectorizer.vectorize(word).values()
                        f_v, t_v = f_v.to(args.device), t_v.to(args.device)
                        lengths = torch.LongTensor([lengths])
                        
                        hidden = model.init_hidden(1, args.device)
                        
                        _, out_cat = ensemble(f_v.unsqueeze(0),
                                              lengths,
                                              hidden,
                                              args.drop_p)
                        
                        target = torch.LongTensor([cat]).to(args.device)
                        
                        loss = F.cross_entropy(out_cat.unsqueeze(0), target, reduction='sum')
                        
                        loss.backward()
                        optimizer.step()
                        
                        _, preds = torch.topk(out_cat, k = 4)
                        acc_cat = ((cat in preds.to('cpu').numpy()) * 100)
                        
                        res['dataset'].append(category)
                        res['prob'].append(end)
                        res['run'].append(run)
                        res['word'].append(word)
                        res['Group'].append(grp)
                        res['label'].append(lab)
                        res['block'].append(it + 1)
                        res['trl'].append(tr+1)
                        res['loss'].append(loss.item())
                        res['type'].append('reco')
                        res['acc'].append(acc_cat)
                        res['response'].append('')
                        
                    # Production task
                    for word, cat, lab in zip(exp_words.data, exp_words.cat, exp_words.label):
                        idxs = []
                        for i, (f_v, t_v) in vectorizer.vectorize_single_char(word):
                            optimizer.zero_grad()
                            f_v, t_v = f_v.to(args.device), t_v.to(args.device)
                            hidden = ensemble.cloud.init_hidden(1, args.device)
                            
                            out_letters, _ = ensemble(f_v.unsqueeze(0), 
                                                            torch.LongTensor([i+1]),
                                                            hidden, 
                                                            args.drop_p)
                            
                            loss = F.cross_entropy(out_letters[-1].unsqueeze(0), t_v,
                                                    ignore_index=mask_index,
                                                    reduction='sum')
                            
                            loss.backward()
                            optimizer.step()
                            
                            _, idx = torch.max(F.softmax(out_letters[-1].detach().to('cpu'), dim=0), 0)
                            idxs.append(idx.item())
                        
                        prod_word = vectorizer.decode(idxs)
                        acc_letters = score_word(prod_word, word)
                        
                        res['dataset'].append(category)
                        res['prob'].append(end)
                        res['run'].append(run)
                        res['word'].append(word)
                        res['Group'].append(grp)
                        res['label'].append(lab)
                        res['block'].append(it + 1)
                        res['trl'].append(tr+1)
                        res['loss'].append(loss.item())
                        res['type'].append('prod')
                        res['acc'].append(acc_letters)
                        res['response'].append(prod_word)

res = pd.DataFrame(res)

#res.to_csv('results/simulation_results.csv', index=False, encoding='utf-8')

#%%
sns.set(style='whitegrid')

g = sns.catplot(x='block', y='acc', hue='Group', hue_order=['MONO', 'ES-EN', 'ES-EU'],
                col='label', col_order=['ES-', 'ES+'], row='type', row_order=['reco', 'prod'],
                data=res, kind='point', ci=99)
g.set(ylim=(0, 100))