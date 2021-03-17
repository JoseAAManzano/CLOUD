# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 09:53:03 2021

@author: josea
"""

# %% Imports

# Utilities
import utils
import torch
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import torch.nn as nn
import torch.nn.functional as F

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', context='paper', palette='colorblind', font_scale=1.5)

# Misc
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

# %% Model extension to predict category

class LabelPredictor(nn.Module):
    def __init__(self, n_in, n_out):
        super(LabelPredictor, self).__init__()
        self.fc = nn.Linear(n_in, n_out)
    
    def forward(self, x):
        return self.fc(x)
    

# %% Use test words to mimic learning during the blocks
utils.set_all_seeds(args.seed, args.device)    
    
eval_words = pd.read_csv(args.csv + 'EXP_WORDS.csv')

vectorizer = utils.Vectorizer.from_df(eval_words)

mask_index = vectorizer.data_vocab.PAD_idx

reps_per_block = 2
topk = 4

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
                               f"{m_name}/{m_name}_{run}_threshold_val_35.pt")
            model.to(args.device)
            model.train()
            
            lp = LabelPredictor(args.hidden_dims, 48)
            lp.to(args.device)
            lp.train()
            
            optimizer = torch.optim.Adam(
                    lp.parameters(),
                    lr=args.learning_rate)
            
            optimizer_letters = torch.optim.Adam(
                    model.parameters(),
                    lr=args.learning_rate)
                        
            for it in range(5):
                for tr in range(reps_per_block):
                    
                    # Recognition task
                    exp_words = exp_words.sample(frac=1., replace=False)
                    for word, cat, lab in zip(exp_words.data, exp_words.cat, exp_words.label):
                        optimizer.zero_grad()
                        rep = torch.zeros(args.hidden_dims).to(args.device)
                        for i, (f_v, t_v) in vectorizer.vectorize_single_char(word):
                            f_v, t_v = f_v.to(args.device), t_v.to(args.device)
                            hidden = model.init_hidden(1, args.device)
                            
                            _, out_rnn, _ = model(f_v.unsqueeze(0), 
                                                            torch.LongTensor([i+1]),
                                                            hidden, 
                                                            args.drop_p)
                            
                            rep += torch.flatten(out_rnn.squeeze(0)[-1])

                        rep /= i+1
                        
                        out_cat = lp(rep)
                        
                        target = torch.LongTensor([cat]).to(args.device)
                        
                        loss = F.cross_entropy(out_cat.unsqueeze(0), target, reduction='sum')
                        
                        loss.backward()
                        optimizer.step()
                        
                        _, preds = torch.topk(F.log_softmax(out_cat, dim=0), k = topk)
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
                    exp_words = exp_words.sample(frac=1., replace=False)
                    for word, cat, lab in zip(exp_words.data, exp_words.cat, exp_words.label):
                        idxs = []
                        for i, (f_v, t_v) in vectorizer.vectorize_single_char(word):
                            optimizer_letters.zero_grad()
                            f_v, t_v = f_v.to(args.device), t_v.to(args.device)
                            hidden = model.init_hidden(1, args.device)
                            
                            out_letters, _, _ = model(f_v.unsqueeze(0), 
                                                            torch.LongTensor([i+1]),
                                                            hidden, 
                                                            args.drop_p)
                            
                            loss = F.cross_entropy(out_letters[-1].unsqueeze(0), t_v,
                                                    ignore_index=mask_index,
                                                    reduction='sum')
                            
                            loss.backward()
                            optimizer_letters.step()
                            
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
            
            # Save a trained version of the model after the 5 blocks
            #torch.save(model, f'models/{m_name}/{m_name}_{run}_trained.pt')

res = pd.DataFrame(res)

#res.to_csv('results/simulation_results.csv', index=False, encoding='utf-8')

g = sns.catplot(x='block', y='acc', hue='Group', hue_order=['MONO', 'ES-EN', 'ES-EU'],
                col='label', col_order=['ES-', 'ES+'], row='type', row_order=['reco', 'prod'],
                data=res, kind='point', ci=99)
g.set(ylim=(0, 100))
plt.show()

