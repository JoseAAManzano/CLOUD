# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 13:48:18 2020

@author: josea
"""

# %% Imports

# Utilities
import utils
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', context='paper', palette='pastel', font_scale=1.5)

# Misc
from argparse import Namespace
from collections import defaultdict
from scipy.stats import ttest_ind

# Readout prediction
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import metrics as mtr


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

#%% Model comparison
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

            model_file = args.model_save_file + f"{m_name}/{m_name}_{run}_threshold_val_35.pt"
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

# Model comparison plot
plt.figure(figsize=(6, 9))
ax = sns.catplot(data=mc, x='Set', y='Accuracy', hue='Version',
                 hue_order=['MONO', 'ES-EN', 'ES-EU'],
                 col='Language', kind='bar', ci=99)
plt.show()


# %% Hidden representation for each time-point for each word
# First is necessary to run the readout.py file to produce the representations
hidd_cols = [f"hid_{str(i+1)}" for i in range(args.hidden_dims)]

res = defaultdict(list)

metrics = {'Accuracy': mtr.accuracy_score, 'F1': mtr.f1_score,
           'ROC_AUC': mtr.roc_auc_score}

def pred_scores(X_train, X_test, y_train, y_test, clf, metrics):
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    
    tmp = defaultdict(float)
    for met, func in metrics.items():
        if met == 'ROC_AUC':
            tmp[met] = func(y_test, preds, average='weighted')
        else:
            tmp[met] = func(y_test, preds)
    return tmp

for data, category in zip(args.datafiles, args.modelfiles):
    for prob in args.probs:
        end = f"{prob:02}-{100-prob:02}"
        m_name = f"{category}_{end}"
        
        test_dataset = pd.DataFrame()
        val_dataset = pd.DataFrame()
        for run in range(args.n_runs):
            hdn = pd.read_json(f"hidden/val_hidden_{m_name}_{run}.json",encoding='utf-8')
            val_dataset = pd.concat([val_dataset, hdn], axis=0, ignore_index=True)
            
            hdn = pd.read_json(f"hidden/test_hidden_{m_name}_{run}.json",encoding='utf-8')
            test_dataset = pd.concat([test_dataset, hdn], axis=0, ignore_index=True)
        del hdn
            
        for run in range(args.n_runs):
            train_data = val_dataset[val_dataset.run == run]
            test_data = test_dataset[test_dataset.run == run]
            for T in range(train_data.char.max()+1):
                model_data = train_data[train_data.char == T]
                model_test = test_data[test_data.char == T]

                X_train1 = model_data[hidd_cols].values
                y_train1 = model_data.label.values
                                
                X_test = model_test[hidd_cols].values
                y_test = model_test.label.values
                
                lr = LogisticRegression(max_iter=10e9, random_state=args.seed)
                
                res_x1 = pred_scores(X_train1, X_test, y_train1, y_test, lr,
                                     metrics)

                res['dataset'].append(category)
                res['prob'].append(end)
                grp = ''
                if category == 'ESEN':
                    if end == '60-40':
                        grp = 'ES-EN'
                    else:
                        grp = 'MONO'
                else:
                    if end == '60-40':
                        grp = 'ES-EU'
                    else:
                        
                        grp = 'MONO'
                res['group'].append(grp)
                res['run'].append(run)
                res['char'].append(T)
                
                for met, val in res_x1.items():
                    res[met].append(val)

                print(
                    f"{m_name}_{run} char-{T}: {res['Accuracy'][-1]:.2f}")
                
res = pd.DataFrame(res)

res.to_csv('results/backup_readout_prediction.csv', index=False, encoding='utf-8')       

# %% Plots
readout = pd.read_csv('results/backup_readout_prediction.csv', encoding='utf-8')
readout['Version'] = readout.group
readout['AUROC'] = readout['ROC_AUC']

pivot_esen = readout[readout.dataset=='ESEN'].pivot_table(index=['run', 'char'],
                              columns='Version',
                              values='AUROC').reset_index()
for ch in range(readout.char.max() + 1):
    tmp = pivot_esen[pivot_esen.char == ch]
    _, pval = ttest_ind(tmp['ES-EN'], tmp['MONO'])
    st = '*' if pval < 0.05/11 else 'n.s.'
    print(f"ES-EN-{ch}: {st}")

pivot_eseu = readout[readout.dataset=='ESEU'].pivot_table(index=['run', 'char'],
                              columns='Version',
                              values='AUROC').reset_index()
for ch in range(readout.char.max() + 1):
    tmp = pivot_eseu[pivot_eseu.char == ch]
    _, pval = ttest_ind(tmp['ES-EU'], tmp['MONO'])
    st = '*' if pval < 0.05/11 else 'n.s.'
    print(f"ES-EU-{ch}: {st}")


g = sns.catplot(x='char', y='AUROC', hue='Version', hue_order=['MONO', 'ES-EN', 'ES-EU'],
                col='dataset',
                data=readout, kind='point', ci=99)
g.set(ylim=(0.49, 1.))
g.axes.flatten()[0].fill([2.5, 8.5, 8.5, 2.5], [0.4,0.4,1,1], 'k', alpha=0.2)
g.axes.flatten()[0].fill([9.5, 10.5, 10.5, 9.5], [0.4,0.4,1,1], 'k', alpha=0.2)
g.axes.flatten()[0].set_ylabel('AUROC', fontsize=15)
g.axes.flatten()[1].fill([2.5, 10.5, 10.5, 2.5], [0.4,0.4,1,1], 'k', alpha=0.2)

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
            
            # ensemble = Ensemble(model, args.hidden_dims, 48)  
            # ensemble.to(args.device)
            # ensemble.train()
            
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
                        # preds = []
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


res = pd.DataFrame(res)

res.to_csv('results/simulation_results.csv', index=False, encoding='utf-8')

g = sns.catplot(x='block', y='acc', hue='Group', hue_order=['MONO', 'ES-EN', 'ES-EU'],
                col='label', col_order=['ES-', 'ES+'], row='type', row_order=['reco', 'prod'],
                data=res, kind='point', ci=99)
g.set(ylim=(0, 100))