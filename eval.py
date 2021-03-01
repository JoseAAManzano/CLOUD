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

#mc.to_csv('results/model_comparison.csv', index=False, encoding='utf-8')

# Model comparison plot
#mc = pd.read_csv('results/model_comparison.csv')
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

# readout = readout[(readout.char > 0) & (readout.char < 11)]

pivot_esen = readout[readout.dataset=='ESEN'].pivot_table(index=['run', 'char'],
                              columns='Version',
                              values='ROC_AUC').reset_index()
for ch in range(readout.char.max() + 1):
    tmp = pivot_esen[pivot_esen.char == ch]
    _, pval = ttest_ind(tmp['ES-EN'], tmp['MONO'])
    st = '*' if pval < 0.05/11 else 'n.s.'
    print(f"ES-EN-{ch}: {st}")

pivot_eseu = readout[readout.dataset=='ESEU'].pivot_table(index=['run', 'char'],
                              columns='Version',
                              values='ROC_AUC').reset_index()
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

pivot = readout.pivot_table(index=['dataset', 'run', 'char'],
                              columns='prob',
                              values='hid_F1').reset_index()
for dtst in ['ESEN', 'ESEU']:
    for ch in range(11):
        tmp = pivot[(pivot.dataset == dtst) & (pivot.char == ch)]
        _, pval = ttest_ind(tmp['60-40'], tmp['100-00'])
        st = '*' if pval < 0.005 else 'n.s.'
        print(f"{dtst}-{ch}: {st}")

g = sns.catplot(x='char', y='hid_F1', hue='prob', row='dataset',
                data=readout, kind='point', palette='Reds', ci='sd')
g.set(ylim=(0.5, 1))
# g.axes.flatten()[0].fill([1.5, 7.5, 7.5, 1.5], [0.5,0.5,1,1], 'k', alpha=0.2)
# g.axes.flatten()[0].set_ylabel('F1', fontsize=15)
# g.axes.flatten()[1].fill([1.5, 9.1, 9.1, 1.5], [0.5,0.5,1,1], 'k', alpha=0.2)
# g.axes.flatten()[1].set_ylabel('F1', fontsize=15)

pivot = readout.pivot_table(index=['dataset', 'run', 'char'],
                              columns='prob',
                              values='cel_ROC_AUC').reset_index()
for dtst in ['ESEN', 'ESEU']:
    for ch in range(11):
        tmp = pivot[(pivot.dataset == dtst) & (pivot.char == ch)]
        _, pval = ttest_ind(tmp['50-50'], tmp['100-00'])
        st = '*' if pval < 0.005 else 'n.s.'
        print(f"{dtst}-{ch}: {st} {pval:.3f}")

g = sns.catplot(x='char', y='cel_ROC_AUC', hue='prob', row='dataset',
                data=readout, kind='point', palette='Blues', ci='sd')
g.axes.flatten()[0].fill([2.5, 7.5, 7.5, 2.5], [0.5,0.5,1,1], 'k', alpha=0.2)
g.axes.flatten()[0].set_ylabel('ROC_AUC', fontsize=15)
g.axes.flatten()[1].fill([2.5, 9.1, 9.1, 2.5], [0.5,0.5,1,1], 'k', alpha=0.2)
g.axes.flatten()[1].set_ylabel('ROC_AUC', fontsize=15)

pivot = readout.pivot_table(index=['dataset', 'run', 'char'],
                              columns='prob',
                              values='cel_F1').reset_index()
for dtst in ['ESEN', 'ESEU']:
    for ch in range(11):
        tmp = pivot[(pivot.dataset == dtst) & (pivot.char == ch)]
        _, pval = ttest_ind(tmp['50-50'], tmp['100-00'])
        st = '*' if pval < 0.005 else 'n.s.'
        print(f"{dtst}-{ch}: {st}")

g = sns.catplot(x='char', y='cel_F1', hue='prob', row='dataset',
                data=readout, kind='point', palette='Blues', ci='sd')
g.set(ylim=(0.5, 1))
g.axes.flatten()[0].fill([2.5, 7.5, 7.5, 2.5], [0.5,0.5,1,1], 'k', alpha=0.2)
g.axes.flatten()[0].set_ylabel('F1', fontsize=15)
g.axes.flatten()[1].fill([2.5, 9.1, 9.1, 2.5], [0.5,0.5,1,1], 'k', alpha=0.2)
g.axes.flatten()[1].set_ylabel('F1', fontsize=15)