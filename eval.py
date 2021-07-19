# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 13:48:18 2020

@author: josea
"""

# %% Imports

# Utilities
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import metrics as mtr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from string import ascii_lowercase
from scipy.stats import ttest_ind
from collections import defaultdict
from argparse import Namespace
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
import json
import numpy as np
import utils
import torch
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

# Plotting
sns.set(style='whitegrid', context='paper', palette='Greys', font_scale=1.5)


palette = ["#27AAE1", "#074C7A", "#666666"]

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


# %% Util functions


def print_from(df, col, d1_l, d1_h, d2_l, d2_h):
    return df[(df['dim1'] >= d1_l) & (df['dim1'] <= d1_h) &
              (df['dim2'] >= d2_l) & (df['dim2'] <= d2_h)][col]


# %% PLOTS FOR FIGURE 1

model_file = 'models/ESEN_60-40/ESEN_60-40_3_threshold_ldt_85.pt'
dataset = pd.read_csv('data/ESP-ENG.csv')
vectorizer = utils.Vectorizer.from_df(dataset)
word = 'model'

model = torch.load(model_file)
model.to('cpu')
model.eval()

utils.set_all_seeds(args.seed, 'cpu')

# %% Plot character probabilities
letters = list(ascii_lowercase) + ['#']
top_k = 5
tmp = np.zeros((top_k, len(word)+1))
annot = [['' for _ in range(len(word)+1)] for _ in range(top_k)]

for i, (f_v, t_v) in vectorizer.vectorize_single_char(word):
    f_v, t_v = f_v.to('cpu'), t_v.to('cpu')

    hidden = model.init_hidden(1)

    out, out_rnn, hidden = model(f_v.unsqueeze(
        0), torch.LongTensor([i+1]), hidden, max_length=i+1)

    dist = torch.flatten(out[-1, :].detach())

    dist = dist[:-1]

    prb = F.softmax(dist, dim=0)

    ret = prb.numpy()

    sorted_probs = np.sort(ret)[::-1][:top_k]
    argsort_probs = np.argsort(ret)[::-1][:top_k]
    for j in range(top_k):
        tmp[j, i] = sorted_probs[j]
        annot[j][i] = f"{letters[argsort_probs[j]].upper()}\n\n{sorted_probs[j]:.2f}"

plt.figure(figsize=(8, 6))
ax = sns.heatmap(data=tmp, cmap='Blues', annot=annot, fmt='',
            xticklabels=list(word.upper()) + ['#'],
            cbar_kws=dict(use_gridspec=False,location="left"),
            vmin=0., vmax=0.6)
ax.set(xticklabels=[], yticklabels=[], xlabel=None, ylabel=None)
plt.show()


# %% Plot word CLOUD
hidd_cols = [f"hid_{i+1}" for i in range(model.rnn.hidden_size)]

hidden_file = model_file.split('/')[-1].split('.')[0]

try:
    hidden_rep = pd.read_json(f"hidden/hidden_repr_{hidden_file}.json",
                              encoding='utf-8')
    print("File found. Loading representatitons from file.")
except:
    print("File not found. Creating hidden representation file. This will take a while.")
    hidd = defaultdict(list)
    for w, l in zip(dataset.data, dataset.label):
        for i, (f_v, t_v) in vectorizer.vectorize_single_char(w):
            f_v, t_v = f_v.to('cpu'), t_v.to('cpu')
            hidden = model.init_hidden(1)
            _, out_rnn, _ = model(f_v.unsqueeze(
                0), torch.LongTensor([i+1]), hidden, max_length=i+1)
            hidd['Word'].append(w)
            hidd['Char'].append(i)
            hidd['Language'].append(l)
            hidd['Length'].append(len(w))
            hid = torch.flatten(out_rnn.squeeze(
                0)[-1].detach()).to('cpu').numpy()
            for k, v in zip(hidd_cols, hid):
                hidd[k].append(float(v))
    print('Generating backup file for future use.')
    with open(f"hidden/hidden_repr_{hidden_file}.json", 'w',
              encoding='utf-8') as f:
        json.dump(hidd, f)
    hidden_rep = pd.DataFrame(hidd)
    del hidd

# Add the representation of the word if it's not in the dataset
if word not in list(dataset.data):
    added = defaultdict(list)
    for i, (f_v, t_v) in vectorizer.vectorize_single_char(w):
        f_v, t_v = f_v.to('cpu'), t_v.to('cpu')
        hidden = model.init_hidden(1)
        _, out_rnn, _ = model(f_v.unsqueeze(
            0), torch.LongTensor([i+1]), hidden, max_length=i+1)
        added['Word'].append(w)
        added['Char'].append(i)
        added['Language'].append("ENG")
        added['Length'].append(len(w))
        hid = torch.flatten(out_rnn.squeeze(0)[-1].detach()).to('cpu').numpy()
        for k, v in zip(hidd_cols, hid):
            added[k].append(float(v))

    hidden_rep = pd.concat([hidden_rep, pd.DataFrame(added)], axis=0)
    del added

# Sum representations by word and language
tmp = hidden_rep.drop(['Char'], axis=1)
df = pd.pivot_table(tmp, index=['Word', 'Length', 'Language'], values=hidd_cols,
                    aggfunc=np.mean).reset_index()
df['Language'] = df.Language.apply(lambda x: x[:-1])

df[hidd_cols] = StandardScaler().fit_transform(df[hidd_cols].values)

# Compress hidden representations for plotting using PCA and TSNE
print('Reducing the dimensionality for plotting. This will take a while.')

pca = PCA(n_components=50)

pca_res = pca.fit_transform(df[hidd_cols])

tsne = TSNE(n_components=2, perplexity=100, n_jobs=-1, random_state=args.seed)

df[['dim1', 'dim2']] = tsne.fit_transform(pca_res)

df['Language'] = df.Language.map({'ES':'SP','EN':'EN'})

# Plot the entire dataset
sns.set(style='white')
ax = sns.jointplot(x='dim1', y='dim2', kind='scatter',
                   hue='Language', hue_order=['SP', 'EN'],
                   data=df, alpha=0.8, space=0.2, palette=["#666666", "#27AAE1"],
                   xlim=(-80, 80), ylim=(-80, 80), s=2, markers=['o', 'x'])
ax.ax_joint.set(xticklabels=[], yticklabels=[], xlabel=None, ylabel=None)
plt.show()

list1 = [x for x in list(print_from(df, 'Word', 35, 45, 30, 40)) if len(x) < 6]
list2 = [x for x in list(print_from(
    df, 'Word', -60, -40, -20, 5)) if len(x) < 7]

# Compute the distance from all words to the selected word
print("Computing similarity")
word_rep = df[df.Word == word][hidd_cols].values[0]

dists = []
for i, row in df.iterrows():
    if (i+1) % 1000 == 0:
        print(f"{i+1} words processed.")
    dists.append(word_rep.dot(row[hidd_cols].values) /
                 (np.linalg.norm(word_rep) *
                  np.linalg.norm(row[hidd_cols].values))
                 )

df['dist'] = dists

# Get most similar and dissimilar words
top = df.sort_values(by='dist', ascending=False)[:16]

def jitter(df, col):
    return df[col] + np.random.randn(len(df)) * (1 * (max(df[col]) - min(df[col])))


top['dim2_j'] = jitter(top, 'dim2')
top['dim1_j'] = jitter(top, 'dim1')


def label_points(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        if str(point['val']) == word:
            ax.text(point['x']-.02, point['y']+.02, str(point['val']),
                    bbox=dict(ec='red', fc='w', alpha=0.7), fontsize=25)
        else:
            ax.text(point['x']-.02, point['y']+.02,
                    str(point['val']), fontsize=20)


plt.figure(figsize=(7, 5))
ax = sns.scatterplot(x='dim1_j', y='dim2_j', hue=top.Language.tolist(), palette=["#666666", "#27AAE1"], data=top,
                     hue_order=['SP', 'EN'], s=35)
ax.set(xticklabels=[], yticklabels=[], xlabel=None, ylabel=None)
sns.despine()
label_points(top['dim1_j'], top['dim2_j'], top.Word, ax)
ax.legend(loc='upper right')
plt.show()


# %% PLOTS FOR FIGURE 2 (DEPRECATED)

# Model comparison
tmp = defaultdict(list)
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
                cat = 'SP-EN'
            else:
                cat = 'SP-BQ'

            model_file = args.model_save_file + \
                f"{m_name}/{m_name}_{run}_threshold_ldt_85.pt"
            print(f"\nSave file: {data}: {model_file}\n")

            model = torch.load(model_file)
            model.eval()

            df = pd.read_csv(args.csv + data)
            vectorizer = utils.Vectorizer.from_df(df)

            train_acc = utils.lexical_decision_task(args, model, vectorizer,
                                              split='train', log=True)
            val_acc = utils.lexical_decision_task(args, model, vectorizer,
                                              split='val', log=True)
            test_acc = utils.lexical_decision_task(args, model, vectorizer,
                                              split='test', log=True)
            tmp['Group'].append(category)
            tmp['run'].append(run)
            tmp['Version'].append(cat)
            tmp['Train_Accuracy'].append(train_acc)
            tmp['Test_Accuracy'].append((val_acc + test_acc)/2)

res = pd.DataFrame(tmp)

# mc = res

# langs = [x[-2:].upper() + "_" + y.split('.')[0].split('-')[1]
#          for x, y in zip(mc['Split'], mc['dataset'])]

# mc['lang'] = langs
# mc['Set'] = mc['Split'].map(lambda x: x[:-3].upper())

# mc = mc[mc.lang != 'L1_EUS']
# mc['Language'] = mc['lang'].map({'L1_ENG': 'Spanish (ES)',
#                                  'L2_ENG': 'English (EN)',
#                                  'L2_EUS': 'Basque (EU)'})

res.to_csv('results/model_comparison.csv', index=False, encoding='utf-8')

# Calculate empirical chance-score
from cloudmodel import CLOUD

emp_train = []
emp_test = []
for i in range(20):
    print(i)
    model = CLOUD(
                char_vocab_size=len(vectorizer.data_vocab),
                n_embedd=args.embedding_dim,
                n_hidden=args.hidden_dims,
                n_layers=args.n_rnn_layers,
                pad_idx=vectorizer.data_vocab.PAD_idx
            )
    
    train_acc = utils.lexical_decision_task(args, model, vectorizer,
                                      split='train', log=True)
    val_acc = utils.lexical_decision_task(args, model, vectorizer,
                                      split='val', log=True)
    test_acc = utils.lexical_decision_task(args, model, vectorizer,
                                      split='test', log=True)
    emp_train.append(train_acc)
    emp_test.append((val_acc + test_acc)/2)
    
print(np.mean(emp_train), np.std(emp_train))
print(np.mean(emp_test), np.std(emp_test))


# %% PLOTS FOR NEW FIGURE 2

# Hidden representation for each time-point for each word
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
    
    prob_corr = 0
    pred_proba = clf.predict_proba(X_test)
    for i, l in enumerate(y_test):
        prob_corr += pred_proba[i, l]
    tmp['Probability'] = prob_corr/len(X_test)
    return tmp


for data, category in zip(args.datafiles, args.modelfiles):
    for prob in args.probs:
        end = f"{prob:02}-{100-prob:02}"
        m_name = f"{category}_{end}"

        test_dataset = pd.DataFrame()
        val_dataset = pd.DataFrame()
        for run in range(args.n_runs):
            hdn = pd.read_json(
                f"hidden/val_hidden_{m_name}_{run}.json", encoding='utf-8')
            val_dataset = pd.concat(
                [val_dataset, hdn], axis=0, ignore_index=True)

            hdn = pd.read_json(
                f"hidden/test_hidden_{m_name}_{run}.json", encoding='utf-8')
            test_dataset = pd.concat(
                [test_dataset, hdn], axis=0, ignore_index=True)
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
                        grp = 'SP-EN'
                    else:
                        grp = 'MONO'
                else:
                    if end == '60-40':
                        grp = 'SP-BQ'
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

# Plots
res['Version'] = res.group

pivot_esen = res[res.dataset == 'ESEN'].pivot_table(index=['run', 'char'],
                                                            columns='Version',
                                                            values='Probability').reset_index()
for ch in range(res.char.max() + 1):
    tmp = pivot_esen[pivot_esen.char == ch]
    _, pval = ttest_ind(tmp['SP-EN'], tmp['MONO'])
    st = '*' if pval < 0.05/11 else 'n.s.'
    print(f"SP-EN-{ch}: {st} {pval*11:.5f}")

pivot_eseu = res[res.dataset == 'ESEU'].pivot_table(index=['run', 'char'],
                                                            columns='Version',
                                                            values='Probability').reset_index()
for ch in range(res.char.max() + 1):
    tmp = pivot_eseu[pivot_eseu.char == ch]
    _, pval = ttest_ind(tmp['SP-BQ'], tmp['MONO'])
    st = '*' if pval < 0.05/11 else 'n.s.'
    print(f"ES-EU-{ch}: {st} {pval*11:.5f}")

sns.set(style='white', font_scale=1.2)

g = sns.catplot(x='char', y='Probability', hue='Version', hue_order=['SP-EN', 'SP-BQ', 'MONO'],
                col='dataset', palette=["#27AAE1", "#074C7A", "#666666"], markers=['o', 'x', '*'],
                data=res, kind='point', ci=99, scale=1.2)
g.set(ylim=(0.48, 1.))
g.axes.flatten()[0].set_xlabel('Character (time-step)')
g.axes.flatten()[1].set_xlabel('Character (time-step)')
g.axes.flatten()[0].set_title('Spanish-English')
g.axes.flatten()[1].set_title('Spanish-Basque')
g.axes.flatten()[0].set_ylabel('Probability Correct \u00B1 95% CI')
g.axes.flatten()[0].axhline(0.5, color='k', linestyle='--', lw=3)
g.axes.flatten()[1].axhline(0.5, color='k', linestyle='--', lw=3)
g.axes.flatten()[0].fill([2.5, 10.5, 10.5, 2.5], [0.4, 0.4, 1, 1], 'k', alpha=0.2)

g.axes.flatten()[1].fill([2.5, 10.5, 10.5, 2.5],
                         [0.4, 0.4, 1, 1], 'k', alpha=0.2)


