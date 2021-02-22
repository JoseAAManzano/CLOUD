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
    n_runs=5,  # How many versions of the models to train
    # Model hyperparameters
    embedding_dim=32,
    hidden_type='LSTM',
    hidden_dims=128,
    n_rnn_layers=1,
    drop_p=0.4,
    # Training hyperparameters
    n_epochs=100,
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
        x1, hid = self.cloud(x, X_lengths, hidden, drop_rate)
        inp2 = torch.flatten(hid[0])
        x2 = self.out(F.dropout(inp2, drop_rate))
        return x1, x2



# %% Use test words to mimic learning during the blocks
utils.set_all_seeds(args.seed, args.device)    
    
eval_words = pd.read_csv(args.csv + 'EXP_WORDS.csv')

vectorizer = utils.Vectorizer.from_df(eval_words)

mask_index = vectorizer.data_vocab.PAD_idx

exp_words = eval_words
exp_words = exp_words.sample(frac=1., replace=False)
exp_words['cat'] = range(48)

reps_per_block = 2

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
    return score if score > 0 else 0

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
            model = torch.load(args.model_save_file +
                               f"{args.hidden_type}/{args.hidden_dims}/{m_name}/{m_name}_{run}_threshold_val_34.pt")

            ensemble = Ensemble(model, args.hidden_dims, 48)
            ensemble.to(args.device)
            
            optimizer = torch.optim.Adam(
                    ensemble.parameters(),
                    lr=args.learning_rate)
            
            # optimizer_cat = torch.optim.Adam(
            #         ensemble.parameters(),
            #         lr=args.learning_rate)
            
            # optimizer_letters = torch.optim.Adam(
            #         ensemble.parameters(),
            #         lr=args.learning_rate)


            for it in range(5):
                for tr in range(reps_per_block):
                    exp_words = exp_words.sample(frac=1., replace=False)
                    
                    # Recognition task
                    ensemble.train()
                    for word, cat, lab in zip(exp_words.data, exp_words.cat, exp_words.label):
                        preds = []
                        idxs = []
                        for i, (f_v, t_v) in vectorizer.vectorize_single_char(word):
                            optimizer.zero_grad()
                            # vector_dict = vectorizer.vectorize(word)
                            # f_v, t_v, lengths = vector_dict.values()
                            f_v, t_v = f_v.to(args.device), t_v.to(args.device)
    
                            hidden = ensemble.cloud.init_hidden(1, args.device)
    
                            out_letters, out_cat = ensemble(f_v.unsqueeze(0),
                                                              torch.LongTensor([i+1]),
                                                              hidden,
                                                              args.drop_p)
                            
                            target = torch.LongTensor([cat]).to(args.device)
                                                        
                            loss1 = F.cross_entropy(out_cat.unsqueeze(0), target, reduction='sum')
                            loss2 = F.cross_entropy(out_letters[-1].unsqueeze(0), t_v,
                                                   ignore_index=mask_index,
                                                   reduction='sum')
                            
                            loss = loss1+loss2
                            
                            loss.backward()
                            optimizer.step()
                            
                            _, pred_cat = torch.max(F.log_softmax(out_cat, dim=0), dim=0)
                            preds.append(pred_cat.item())
                            
                            _, idx = torch.max(F.softmax(out_letters[-1].detach().to('cpu'), dim=0), 0)
                            idxs.append(idx.item())
                        
                        if tr == reps_per_block-1:
                            pred_cat = max(set(preds), key=preds.count)
                            acc_cat = ((cat == pred_cat) * 100)
                            
                            prod_word = vectorizer.decode(idxs)
                            acc_letters = score_word(prod_word, word)
                            
                            for tp in ['reco', 'prod']:            
                                res['dataset'].append(category)
                                res['prob'].append(end)
                                res['run'].append(run)
                                res['word'].append(word)
                                res['Group'].append(grp)
                                res['label'].append(lab)
                                res['block'].append(it + 1)
                                res['trl'].append(tr+1)
                                res['loss'].append(loss.item())
                                if tp == 'reco':
                                    res['type'].append('reco')
                                    res['acc'].append(acc_cat)
                                    res['response'].append('')
                                else:
                                    res['type'].append('prod')
                                    res['acc'].append(acc_letters)
                                    res['response'].append(prod_word)

                        
                    # # Production task
                    # for word, cat, lab in zip(exp_words.data, exp_words.cat, exp_words.label):
                        
                    #     for i, (f_v, t_v) in vectorizer.vectorize_single_char(word):
                    #         optimizer_letters.zero_grad()
                    #         f_v, t_v = f_v.to(args.device), t_v.to(args.device)
                    #         hidden = ensemble.cloud.init_hidden(1, args.device)
                            
                    #         out_letters, _ = ensemble(f_v.unsqueeze(0), 
                    #                                         torch.LongTensor([i+1]),
                    #                                         hidden, 
                    #                                         args.drop_p)
                            
                    #         loss = F.cross_entropy(out_letters[-1].unsqueeze(0), t_v,
                    #                                ignore_index=mask_index,
                    #                                reduction='sum')
                            
                    #         loss.backward()
                    #         optimizer_letters.step()
                            
                    #         _, idx = torch.max(F.softmax(out_letters[-1].detach().to('cpu'), dim=0), 0)
                    #         idxs.append(idx.item())
                        
                    #     prod_word = vectorizer.decode(idxs)
                    #     acc_letters = score_word(prod_word, word)
                    #     res['dataset'].append(category)
                    #     res['prob'].append(end)
                    #     res['run'].append(run)
                    #     res['word'].append(word)
                    #     res['Group'].append(grp)
                    #     res['label'].append(lab)
                    #     res['block'].append(it + 1)
                    #     res['trl'].append(tr+1)
                    #     res['loss'].append(loss.item())
                    #     res['type'].append('prod')
                    #     res['acc'].append(acc_letters)
                    #     res['response'].append(prod_word)

res = pd.DataFrame(res)

#res.to_csv('results/simulation_results.csv', index=False, encoding='utf-8')

sns.set(style='white')

g = sns.catplot(x='block', y='acc', hue='Group', hue_order=['MONO', 'ES-EN', 'ES-EU'],
                col='label', col_order=['ES-', 'ES+'], row='type', row_order=['reco', 'prod'],
                data=res, kind='point', ci=99)
g.set(ylim=(0, 100))



#%%# esm = eval_words[eval_words.label == 'ES+']


            # for it in range(5):
            #     for tr in range(reps_per_block):
            #         esm = esm.sample(frac=1., replace=False)
            #         esp = esp.sample(frac=1., replace=False)
                    
            #         ensemble_esp.train()
            #         for word, cat, lab in zip(esp.data, esp.cat, esp.label):
            #             optim_esp.zero_grad()
            #             vector_dict = vectorizer.vectorize(word)
            #             f_v, t_v, lengths = vector_dict.values()
            #             f_v, t_v = f_v.to(args.device), t_v.to(args.device)
            #             lengths = torch.LongTensor([lengths]).to(args.device)

            #             hidden = ensemble_esp.cloud.init_hidden(1, args.device)

            #             out_letters, out_cat = ensemble_esp(f_v.unsqueeze(0), lengths, hidden, args.drop_p)
                        
            #             target = torch.LongTensor([cat]).to(args.device)
                        
            #             loss1 = F.cross_entropy(out_letters, t_v, ignore_index=mask_index, reduction='sum')
            #             loss2 = F.cross_entropy(out_cat.unsqueeze(0), target, reduction='sum')

            #             loss_esp = loss1 + loss2
            #             loss_esp.backward()
            #             optim_esp.step()
                      
            #         ensemble_esm.train()
            #         for word, cat, lab in zip(esm.data, esm.cat, esm.label):
            #             optim_esm.zero_grad()
            #             vector_dict = vectorizer.vectorize(word)
            #             f_v, t_v, lengths = vector_dict.values()
            #             f_v, t_v = f_v.to(args.device), t_v.to(args.device)
            #             lengths = torch.LongTensor([lengths]).to(args.device)

            #             hidden = ensemble_esm.cloud.init_hidden(1, args.device)

            #             out_letters, out_cat = ensemble_esm(f_v.unsqueeze(0), lengths, hidden, args.drop_p)
                        
            #             target = torch.LongTensor([cat]).to(args.device)
                        
            #             loss1 = F.cross_entropy(out_letters, t_v, ignore_index=mask_index, reduction='sum')
            #             loss2 = F.cross_entropy(out_cat.unsqueeze(0), target, reduction='sum')

            #             loss_esm = loss1 + loss2
            #             loss_esm.backward()
            #             optim_esm.step()
                    
            #     ensemble_esp.eval()
            #     for word, cat, lab in zip(esp.data, esp.cat, esp.label):
            #         vector_dict = vectorizer.vectorize(word)
            #         f_v, t_v, lengths = vector_dict.values()
            #         f_v, t_v = f_v.to(args.device), t_v.to(args.device)
            #         lengths = torch.LongTensor([lengths]).to(args.device)

            #         hidden = ensemble_esp.cloud.init_hidden(1, args.device)

            #         out_letters, out_cat = ensemble_esp(f_v.unsqueeze(0), lengths, hidden)
                
            #         _, pred_cat = torch.topk(F.log_softmax(out_cat, dim=0), k=6)
            #         pred_cat = pred_cat.detach().to('cpu').numpy()
            #         acc_cat = (cat in pred_cat) * 100
                    
            #         _, idxs = torch.max(F.softmax(out_letters.detach().to('cpu'), dim=1), 1)
                    
            #         prod_word = vectorizer.decode(idxs.numpy())
            #         acc_letters = score_word(prod_word, word)
                    
            #         for tp in ['reco', 'prod']:
            #             res['dataset'].append(category)
            #             res['prob'].append(end)
            #             res['run'].append(run)
            #             res['word'].append(word)
            #             res['Group'].append(grp)
            #             res['label'].append(lab)
            #             res['block'].append(it + 1)
            #             res['trl'].append(tr+1)
            #             res['loss'].append(loss_esp.item())
            #             if tp == 'prod':
            #                 res['type'].append('prod')
            #                 res['acc'].append(acc_letters)
            #                 res['response'].append(prod_word)
            #             else:
            #                 res['type'].append('reco')
            #                 res['acc'].append(acc_cat)
            #                 res['response'].append('')
                
            #     ensemble_esm.eval()
            #     for word, cat, lab in zip(esm.data, esm.cat, esm.label):
            #         vector_dict = vectorizer.vectorize(word)
            #         f_v, t_v, lengths = vector_dict.values()
            #         f_v, t_v = f_v.to(args.device), t_v.to(args.device)
            #         lengths = torch.LongTensor([lengths]).to(args.device)

            #         hidden = ensemble_esm.cloud.init_hidden(1, args.device)

            #         out_letters, out_cat = ensemble_esm(f_v.unsqueeze(0), lengths, hidden)
                
            #         _, pred_cat = torch.topk(F.log_softmax(out_cat, dim=0), k=6)
            #         pred_cat = pred_cat.detach().to('cpu').numpy()
            #         acc_cat = (cat in pred_cat) * 100
                    
            #         _, idxs = torch.max(F.softmax(out_letters.detach().to('cpu'), dim=1), 1)
                    
            #         prod_word = vectorizer.decode(idxs.numpy())
            #         acc_letters = score_word(prod_word, word)
                    
            #         for tp in ['reco', 'prod']:
            #             res['dataset'].append(category)
            #             res['prob'].append(end)
            #             res['run'].append(run)
            #             res['word'].append(word)
            #             res['Group'].append(grp)
            #             res['label'].append(lab)
            #             res['block'].append(it + 1)
            #             res['trl'].append(tr+1)
            #             res['loss'].append(loss_esm.item())
            #             if tp == 'prod':
            #                 res['type'].append('prod')
            #                 res['acc'].append(acc_letters)
            #                 res['response'].append(prod_word)
            #             else:
            #                 res['type'].append('reco')
            #                 res['acc'].append(acc_cat)
            #                 res['response'].append('')

# esm['cat'] = range(24)

# for data, category in zip(args.datafiles, args.modelfiles):
#     for prob in args.probs:
#         if category == "ESEU_" and prob == 100:
#             continue
#         end = f"{prob:02}-{100-prob:02}"
#         m_name = f"{category}{end}"

#         dataset = utils.TextDataset.load_dataset_and_make_vectorizer(
#             args.csv + data,
#             p=prob / 100, seed=args.seed)
#         vectorizer = dataset.get_vectorizer()

#         for run in range(args.n_runs):
#             print(f"\n{data}: {m_name}_{run}\n")
#             lstm_model = torch.load(args.model_save_file +
#                                     f"{m_name}/{m_name}_{run}.pt")
#             lstm_model.to(args.device)

#             ensemble = Ensemble(lstm_model, args.hidden_dim, ensemble_dims)
#             ensemble.to(args.device)

#             loss_fn = nn.CrossEntropyLoss(
#                 ignore_index=vectorizer.data_vocab.PAD_idx)
#             loss_fn2 = nn.CrossEntropyLoss()

#             if optimizer == 'RMSprop':
#                 optim = torch.optim.RMSprop(
#                     ensemble.parameters(),
#                     lr=lr)
#             if optimizer == 'Adam':
#                 optim = torch.optim.Adam(
#                     ensemble.parameters(),
#                     lr=lr)
#             if optimizer == 'SGD':
#                 optim = torch.optim.SGD(
#                     ensemble.parameters(),
#                     lr=lr)
#             # optim2 = torch.optim.RMSprop(
#             #     clf.parameters(),
#             #     lr=lr)

#             if familiarization:
#                 esm = esm.sample(frac=1)
#                 ensemble.train()
#                 for word, lab, cat in zip(esm.data, esm.label, esm.cat):
#                     optim.zero_grad()
#                     f_v, t_v = vectorizer.vectorize(word)
#                     f_v, t_v = f_v.to(args.device), t_v.to(args.device)

#                     target = torch.LongTensor([cat]).to(args.device)

#                     hidden = ensemble.polychar.initHidden(1, args.device)

#                     out1, out2 = ensemble(f_v.unsqueeze(0), hidden)

#                     l1 = loss_fn(*utils.normalize_sizes(out1, t_v))
#                     l2 = loss_fn2(out2.unsqueeze(0), target)
#                     loss = l1+l2*topk

#                     loss.backward()
#                     optim.step()

#             for it in range(5):
#                 for tr in range(reps_per_block):
#                     esm = esm.sample(frac=1)
#                     ensemble.train()
#                     for word, lab, cat in zip(esm.data, esm.label, esm.cat):
#                         optim.zero_grad()
#                         f_v, t_v = vectorizer.vectorize(word)
#                         f_v, t_v = f_v.to(args.device), t_v.to(args.device)

#                         target = torch.LongTensor([cat]).to(args.device)

#                         hidden = ensemble.polychar.initHidden(1, args.device)

#                         out1, out2 = ensemble(f_v.unsqueeze(0), hidden)

#                         l1 = loss_fn(*utils.normalize_sizes(out1, t_v))
#                         l2 = loss_fn2(out2.unsqueeze(0), target)
#                         loss = l1+l2

#                         loss.backward()
#                         optim.step()

#                         _, pred_label = torch.topk(F.log_softmax(out2, dim=0),
#                                                     k=topk)

#                         pred_label = pred_label.detach().to('cpu').numpy()

#                         acc_label = (cat in pred_label) * 100

#                         acc_letters = utils.compute_accuracy(out1, t_v,
#                                                               vectorizer.data_vocab.PAD_idx)

#                         if prob == args.probs[0]:
#                             if category[:-1] == 'ESEN':
#                                 grp = 'ES-EN'
#                             else:
#                                 grp = 'ES-EU'
#                         else:
#                             grp = 'MONO'

#                         for l in range(2):
#                             res['dataset'].append(category[:-1])
#                             res['prob'].append(end)
#                             res['run'].append(run)
#                             res['word'].append(word)
#                             res['Group'].append(grp)
#                             res['label'].append(lab)
#                             res['block'].append(it + 1)
#                             res['trl'].append(tr+1)
#                             res['loss'].append(loss.item())
#                             if l == 0:
#                                 res['Type'].append('reco')
#                                 res['acc'].append(acc_label)
#                             else:
#                                 res['Type'].append('prod')
#                                 res['acc'].append(acc_letters)

#                     # esm = esm.sample(frac=1)
#                     # lstm_model.train()
#                     # for word, lab, cat in zip(esm.data, esm.label, esm.cat):
#                     #     optim1.zero_grad()

#                     #     f_v, t_v = vectorizer.vectorize(word)
#                     #     f_v, t_v = f_v.to(args.device), t_v.to(args.device)

#                     #     hidden = lstm_model.initHidden(1, args.device)
#                     #     out, hidden = lstm_model(f_v.unsqueeze_(0), hidden)

#                     #     loss = loss_fn(*utils.normalize_sizes(out, t_v))

#                     #     loss.backward()
#                     #     optim1.step()

#                     #     acc_letters = utils.compute_accuracy(out, t_v,
#                     #                                  vectorizer.data_vocab.PAD_idx)

#                     #     if prob == 50:
#                     #         if category[:-1] == 'ESEN':
#                     #             grp = 'ES-EN'
#                     #         else:
#                     #             grp = 'ES-EU'
#                     #     else:
#                     #         grp = 'MONO'

#                     #     res['dataset'].append(category[:-1])
#                     #     res['prob'].append(end)
#                     #     res['run'].append(run)
#                     #     res['word'].append(word)
#                     #     res['Group'].append(grp)
#                     #     res['label'].append(lab)
#                     #     res['block'].append(it + 1)
#                     #     res['trl'].append(tr+1)
#                     #     res['loss'].append(loss.item())
#                     #     res['Type'].append('prod')
#                     #     res['acc'].append(acc_letters)


# res.to_csv('simulation_results.csv', index=False, encoding='utf-8')
# %%
