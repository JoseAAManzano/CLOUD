# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 09:53:03 2021

@author: josea
"""

# %% Imports

# Utilities
from collections import defaultdict
from argparse import Namespace
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import utils
import torch
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'


# Plotting
sns.set(style='whitegrid', context='paper',
        palette='colorblind', font_scale=1.5)

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
    embedding_dim=16,
    hidden_dims=128,
    n_rnn_layers=1,
    drop_p=0.0,
    # Training hyperparameters
    learning_rate=2e-3,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    seed=404
)

utils.set_all_seeds(args.seed, args.device)

# %% Model extension to predict category


class TaskSubsystem(nn.Module):
    def __init__(self, n_in, n_out):
        super(TaskSubsystem, self).__init__()
        self.fc = nn.Linear(n_in, n_out)

    def forward(self, x):
        return self.fc(x)


# %% Use test words to mimic learning during the blocks
utils.set_all_seeds(args.seed, args.device)

eval_words = pd.read_csv(args.csv + 'EXP_WORDS.csv')

vectorizer = utils.Vectorizer.from_df(eval_words)

mask_index = vectorizer.data_vocab.PAD_idx

threshold = 'threshold_ldt_85'

# reps_per_block = 1
# topk = 4


# def score_word(prod, target):
#     score = 0
#     if prod == target:
#         score = 100
#     else:
#         for i, l in enumerate(prod):
#             if i > len(target)-1:
#                 score -= 20
#             elif l == target[i]:
#                 score += 20
#     return score if 0 <= score <= 100 else 0


res = defaultdict(list)
for data, category in zip(args.datafiles, args.modelfiles):
    for prob in args.probs:
        if category == "ESEU" and prob == args.probs[-1]:
            continue
        end = f"{prob:02}-{100-prob:02}"
        m_name = f"{category}_{end}"

        if prob == args.probs[0]:
            if category == 'ESEN':
                grp = 'SP-EN'
            else:
                grp = 'SP-BQ'
        else:
            grp = 'MONO'

        for run in range(args.n_runs):
            print(f"\n{data}: {m_name}_{run}\n")

            exp_words = eval_words
            exp_words = exp_words.sample(frac=1., replace=False)
            exp_words['cat'] = range(48)

            model = torch.load(args.model_save_file +
                                f"{m_name}/{m_name}_{run}_{threshold}.pt")
            model.to(args.device)
            model.train()

            lp = TaskSubsystem(args.hidden_dims, 48)
            lp.to(args.device)
            lp.train()

            optimizer = torch.optim.Adam(
                lp.parameters(),
                lr=args.learning_rate)
            
            optimizer_letters = torch.optim.Adam(
                model.parameters(),
                lr=args.learning_rate)
            
            for it in range(6):
            #RECOGNITION TASK
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

                    # selection = np.random.choice([x for x in range(48) if x != cat],
                    #                              size=3, replace=False)
                    # total = np.append(selection, cat)
                    
                    # tt = torch.LongTensor(total)

                    loss = F.cross_entropy(
                        out_cat.unsqueeze(0), target, reduction='sum')

                    loss.backward()
                    optimizer.step()

                    # _, preds = torch.topk(
                    #     F.log_softmax(out_cat, dim=0), k=topk)
                    # acc_cat = ((cat in preds.to('cpu').numpy()) * 100)

                    selection = np.random.choice([x for x in range(48) if x != cat],
                                                  size=3, replace=False)
                    total = np.append(selection, cat)
                    
                    # #Luce's rule
                    # cats = out_cat.detach().to('cpu').numpy()
                    # k=1
                    
                    # cat_prob = np.exp(k*cats[cat]) / np.sum(np.exp(k*cats[total]))
                    
                    cat_prob = F.softmax(
                        out_cat[total], dim=0).detach().to('cpu').numpy()
                    
                    res['dataset'].append(category)
                    res['prob'].append(end)
                    res['run'].append(run)
                    res['word'].append(word)
                    res['Group'].append(grp)
                    res['label'].append(lab)
                    res['block'].append(it)
                    res['loss'].append(loss.item())
                    res['type'].append('reco')
                    res['acc'].append(cat_prob[-1])
                    res['acc_log'].append(np.log(cat_prob[-1]))
            
            # PRODUCTION TASK
            # model = torch.load(args.model_save_file +
            #                     f"{m_name}/{m_name}_{run}_{threshold}.pt")
            # model.to(args.device)
            # model.train()

            # for it in range(6):
                exp_words = exp_words.sample(frac=1., replace=False)
                for word, cat, lab in zip(exp_words.data, exp_words.cat, exp_words.label):
                    word_prob = 5
                    loss = 0
                    optimizer_letters.zero_grad()
                    for i, (f_v, t_v) in vectorizer.vectorize_single_char(word):
                        f_v, t_v = f_v.to(args.device), t_v.to(args.device)
                        hidden = model.init_hidden(1, args.device)

                        out_letters, _, _ = model(f_v.unsqueeze(0),
                                                    torch.LongTensor([i+1]),
                                                    hidden,
                                                    args.drop_p)

                        loss += F.cross_entropy(out_letters[-1].unsqueeze(0), t_v,
                                                ignore_index=mask_index,
                                                reduction='sum')

                        # _, idx = torch.max(
                        #     F.softmax(out_letters[-1].detach().to('cpu'), dim=0), 0)
                        # idxs.append(idx.item())

                        probs = F.softmax(
                            out_letters[-1], dim=0).detach().to('cpu').numpy()
                        word_prob *= probs[t_v.item()]

                    loss.backward()
                    optimizer_letters.step()
                    # prod_word = vectorizer.decode(idxs)
                    # acc_letters = score_word(prod_word, word)

                    res['dataset'].append(category)
                    res['prob'].append(end)
                    res['run'].append(run)
                    res['word'].append(word)
                    res['Group'].append(grp)
                    res['label'].append(lab)
                    res['block'].append(it)
                    res['loss'].append(loss.item())
                    res['type'].append('prod')
                    res['acc'].append(word_prob)
                    res['acc_log'].append(np.log(word_prob))

            # Save a trained version of the model after the 5 blocks
            torch.save(model, f'models/{m_name}/{m_name}_{run}_{threshold}_trained.pt')

res = pd.DataFrame(res)

res.to_csv('results/simulation_results.csv', index=False, encoding='utf-8')


g = sns.catplot(x='block', y='acc_log', hue='Group', hue_order=['SP-EN', 'SP-BQ', 'MONO'],
                palette=["#27AAE1", "#074C7A", "#666666"],
                col='label', col_order=['ES-', 'ES+'], #row='type', row_order=['reco', 'prod'],
                data=res[(res.type == 'reco') & (res.block > 0)], kind='point', ci=95)
plt.show()

g = sns.catplot(x='block', y='acc_log', hue='Group', hue_order=['SP-EN', 'SP-BQ', 'MONO'],
                palette=["#27AAE1", "#074C7A", "#666666"],
                col='label', col_order=['ES-', 'ES+'], #row='type', row_order=['reco', 'prod'],
                data=res[(res.type == 'prod') & (res.block > 0)], kind='point', ci=95)
plt.show()


# %% Chance-level
from cloudmodel import CLOUD

reco_untrained = []
prod_untrained = []

for i in range(20):
    untrained = CLOUD(
                char_vocab_size=len(vectorizer.data_vocab),
                n_embedd=args.embedding_dim,
                n_hidden=args.hidden_dims,
                n_layers=args.n_rnn_layers,
                pad_idx=vectorizer.data_vocab.PAD_idx
            )
    untrained.to(args.device)
    untrained.eval()
    
    lp = TaskSubsystem(args.hidden_dims, 48)
    lp.to(args.device)
    lp.eval()
    
    for it in range(6):
        exp_words = exp_words.sample(frac=1., replace=False)
        for word, cat, lab in zip(exp_words.data, exp_words.cat, exp_words.label):
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

            cat_prob = F.softmax(
                out_cat, dim=0).detach().to('cpu').numpy()
            cat_prob = cat_prob[cat]
    
            reco_untrained.append(np.log(cat_prob))
    
    # PRODUCTION TASK
    # model = torch.load(args.model_save_file +
    #                     f"{m_name}/{m_name}_{run}_{threshold}.pt")
    # model.to(args.device)
    # model.train()
    
    
    
    # for it in range(6):
        exp_words = exp_words.sample(frac=1., replace=False)
        for word, cat, lab in zip(exp_words.data, exp_words.cat, exp_words.label):
            word_prob = 1
            for i, (f_v, t_v) in vectorizer.vectorize_single_char(word):
                f_v, t_v = f_v.to(args.device), t_v.to(args.device)
                hidden = model.init_hidden(1, args.device)
    
                out_letters, _, _ = model(f_v.unsqueeze(0),
                                            torch.LongTensor([i+1]),
                                            hidden,
                                            args.drop_p)
    
    
                # _, idx = torch.max(
                #     F.softmax(out_letters[-1].detach().to('cpu'), dim=0), 0)
                # idxs.append(idx.item())
    
                probs = F.softmax(
                    out_letters[-1], dim=0).detach().to('cpu').numpy()
                word_prob *= probs[t_v.item()]
    
            prod_untrained.append(np.log(word_prob))
    
print(np.mean(reco_untrained), np.std(reco_untrained))
print(np.mean(prod_untrained), np.std(prod_untrained))