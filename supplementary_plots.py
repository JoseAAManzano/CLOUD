# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 10:03:41 2021

@author: josea
"""


# %% Imports

# Utilities
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import metrics as mtr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
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
import scipy.cluster.hierarchy as sch
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


# %% SUPPLEMENTARY FIGURE 2

# Similarity of word representations to each language
hidd_cols = [f"hid_{str(i+1)}" for i in range(args.hidden_dims)]

similarities = defaultdict(list)
for data, category in zip(args.datafiles, args.modelfiles):
    for prob in args.probs:
        end = f"{prob:02}-{100-prob:02}"
        m_name = f"{category}_{end}"
        print(m_name)

        val_dataset = pd.DataFrame()
        for run in range(args.n_runs):
            hdn = pd.read_json(
                f"hidden/val_hidden_{m_name}_{run}.json", encoding='utf-8')
            val_dataset = pd.concat(
                [val_dataset, hdn], axis=0, ignore_index=True)

            hdn = pd.read_json(
                f"hidden/test_hidden_{m_name}_{run}.json", encoding='utf-8')
            val_dataset = pd.concat(
                [val_dataset, hdn], axis=0, ignore_index=True)
        del hdn

        val_dataset = val_dataset.drop('char', axis=1)
        val_dataset.loc[:, 'len'] = val_dataset.word.map(len)

        df = pd.pivot_table(val_dataset, index=['word', 'len', 'label', 'run'], values=hidd_cols,
                            aggfunc=np.mean).reset_index()
        df.loc[:, 'Language'] = df.label.apply(lambda x: x[:-1])
        df = df.sort_values(by=['Language', 'len'])

        for run in range(args.n_runs):
            langs = list(df.Language.unique())
            for ln in langs:
                tmp = df[(df.run == run) & (df.Language == ln)]
                D = cosine_similarity(
                    tmp[hidd_cols].values, tmp[hidd_cols].values)

                similarities['dataset'].append(category)
                similarities['prob'].append(end)
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
                similarities['Version'].append(grp)
                similarities['run'].append(run)
                l = 'L1' if ln == 'ES' else 'L2'
                similarities['Language'].append(l)
                similarities['Type'].append('Within')
                similarities['avg_dist'].append(np.triu(D, 1).mean())

            tmp = df[df.run == run]
            tmp1 = tmp[tmp.Language == langs[0]]
            tmp2 = tmp[tmp.Language == langs[1]]
            D = cosine_similarity(
                tmp1[hidd_cols].values, tmp2[hidd_cols].values)
            similarities['dataset'].append(category)
            similarities['prob'].append(end)
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
            similarities['Version'].append(grp)
            similarities['run'].append(run)
            similarities['Language'].append('L2')
            similarities['Type'].append('Across')
            similarities['avg_dist'].append(np.triu(D, 1).mean())

            if run == 0:
                print(
                    'Reducing the dimensionality for plotting. This will take a while.')
                tmp[hidd_cols] = StandardScaler().fit_transform(tmp[hidd_cols])

                pca = PCA(n_components=50)

                pca_res = pca.fit_transform(tmp[hidd_cols])

                tsne = TSNE(n_components=2, perplexity=100, n_jobs=-1,
                            random_state=args.seed)

                tmp.loc[:, ['dim1', 'dim2']] = tsne.fit_transform(pca_res)

                if category == 'ESEN':
                    palette = ['C1', 'C0']
                    hue_order = ['ES', 'EN']
                else:
                    palette = ['C1', 'C2']
                    hue_order = ['ES', 'EU']

                ax = sns.jointplot(x='dim1', y='dim2', kind='scatter',
                                   hue='Language', hue_order=hue_order, palette=palette,
                                   data=tmp, alpha=0.8, space=0.1,
                                   xlim=(-70, 70), ylim=(-70, 70), s=5)
                plt.show()

similarities = pd.DataFrame(similarities)
similarities.loc[:, 'Contrast'] = similarities[[
    'Language', 'Type']].agg('_'.join, axis=1)
similarities.loc[:, 'Model'] = similarities[[
    'dataset', 'prob']].agg('_'.join, axis=1)

similarities['Contrast'] = similarities.Contrast.map({'L1_Within': 'Within L1',
                                                      'L2_Within': 'Within L2',
                                                      'L2_Across': 'Across\nLanguages'})

sns.set(style='whitegrid', context='paper',
        palette='colorblind', font_scale=1.8)
g = sns.catplot(x='Contrast', y='avg_dist', order=['Within L1', 'Within L2', 'Across\nLanguages'],
                col='Model', palette='Greys', label='big',
                data=similarities, kind='violin', inner='point')
g.axes.flatten()[0].set_ylabel('Avg. Cosine Similarity', fontsize=18)
g.set(ylim=(0.2, 0.4), xlabel='', title='')
plt.show()

#similarities.to_csv('results/backup_supplementary_similarities.csv', index=False, encoding='utf-8')

# %% SUPPLEMENTARY FIGURE 3
sns.set(style='whitegrid', context='paper',
        palette='colorblind', font_scale=1.5)

# Similarity of Flavian representations to each language
hidd_cols = [f"hid_{str(i+1)}" for i in range(args.hidden_dims)]

eval_words = pd.read_csv(args.csv + 'EXP_WORDS.csv')

vectorizer = utils.Vectorizer.from_df(eval_words)

plot = True

similarities = defaultdict(list)
for data, category in zip(args.datafiles, args.modelfiles):
    for prob in args.probs:
        if prob == 100 and category == 'ESEU':
            continue
        end = f"{prob:02}-{100-prob:02}"
        m_name = f"{category}_{end}"
        print(m_name)

        words_dataset = pd.DataFrame()
        for run in range(args.n_runs):
            hdn = pd.read_json(
                f"hidden/val_hidden_{m_name}_{run}.json", encoding='utf-8')
            words_dataset = pd.concat(
                [words_dataset, hdn], axis=0, ignore_index=True)

            hdn = pd.read_json(
                f"hidden/test_hidden_{m_name}_{run}.json", encoding='utf-8')
            words_dataset = pd.concat(
                [words_dataset, hdn], axis=0, ignore_index=True)
        del hdn

        words_dataset = words_dataset.drop('char', axis=1)
        words_dataset.loc[:, 'len'] = words_dataset.word.map(len)

        df = pd.pivot_table(words_dataset, index=['word', 'len', 'label', 'run'], values=hidd_cols,
                            aggfunc=np.mean).reset_index()
        df.loc[:, 'Language'] = df.label.apply(lambda x: x[:-1])
        df = df.sort_values(by=['Language', 'len'])

        words_dataset = pd.DataFrame()
        for run in range(args.n_runs):
            hdn = pd.read_json(
                f"hidden/val_hidden_{m_name}_{run}_trained.json", encoding='utf-8')
            words_dataset = pd.concat(
                [words_dataset, hdn], axis=0, ignore_index=True)

            hdn = pd.read_json(
                f"hidden/test_hidden_{m_name}_{run}_trained.json", encoding='utf-8')
            words_dataset = pd.concat(
                [words_dataset, hdn], axis=0, ignore_index=True)
        del hdn

        words_dataset = words_dataset.drop('char', axis=1)
        words_dataset.loc[:, 'len'] = words_dataset.word.map(len)

        df_trained = pd.pivot_table(words_dataset, index=['word', 'len', 'label', 'run'], values=hidd_cols,
                                    aggfunc=np.mean).reset_index()
        df_trained['Language'] = df_trained.label.apply(lambda x: x[:-1])
        df_trained = df_trained.sort_values(by=['Language', 'len'])

        del words_dataset

        hidd_repr = defaultdict(list)
        for run in range(args.n_runs):
            model = torch.load(args.model_save_file +
                               f"{m_name}/{m_name}_{run}_threshold_val_35.pt")
            model.to('cpu')
            model.eval()

            for word, lab in zip(eval_words.data, eval_words.label):
                rep = np.zeros(args.hidden_dims)
                for i, (f_v, t_v) in vectorizer.vectorize_single_char(word):
                    hidden = model.init_hidden(1, 'cpu')

                    _, out_rnn, _ = model(f_v.unsqueeze(0),
                                          torch.LongTensor([i+1]),
                                          hidden,
                                          args.drop_p)
                    rep += torch.flatten(out_rnn.squeeze(0)
                                         [-1].detach()).numpy()

                rep /= i+1
                hidd_repr['word'].append(word)
                hidd_repr['len'].append(i)
                hidd_repr['label'].append(lab)
                hidd_repr['run'].append(run)
                for k, v in zip(hidd_cols, rep):
                    hidd_repr[k].append(float(v))
                hidd_repr['Language'].append(lab)

        df = pd.concat([df, pd.DataFrame(hidd_repr)],
                       axis=0, ignore_index=True)

        hidd_repr = defaultdict(list)
        for run in range(args.n_runs):
            model = torch.load(args.model_save_file +
                               f"{m_name}/{m_name}_{run}_trained.pt")
            model.to('cpu')
            model.eval()

            for word, lab in zip(eval_words.data, eval_words.label):
                rep = np.zeros(args.hidden_dims)
                for i, (f_v, t_v) in vectorizer.vectorize_single_char(word):
                    hidden = model.init_hidden(1, 'cpu')

                    _, out_rnn, _ = model(f_v.unsqueeze(0),
                                          torch.LongTensor([i+1]),
                                          hidden,
                                          args.drop_p)
                    rep += torch.flatten(out_rnn.squeeze(0)
                                         [-1].detach()).numpy()

                rep /= i+1
                hidd_repr['word'].append(word)
                hidd_repr['len'].append(i)
                hidd_repr['label'].append(lab)
                hidd_repr['run'].append(run)
                for k, v in zip(hidd_cols, rep):
                    hidd_repr[k].append(float(v))
                hidd_repr['Language'].append(lab)

        df_trained = pd.concat(
            [df_trained, pd.DataFrame(hidd_repr)], axis=0, ignore_index=True)

        mappings = {'ES': 'ES', 'EN': 'L2',
                    'EU': 'L2', 'ES+': 'ES+', 'ES-': 'ES-'}
        df['Language'] = df.Language.map(mappings)
        df_trained['Language'] = df_trained.Language.map(mappings)

        datas = ['Untrained', 'Trained']
        dfs = [df, df_trained]

        for data_label, data_df in zip(datas, dfs):
            for run in range(args.n_runs):
                tmp = data_df[data_df.run == run]

                l1 = tmp[tmp.Language == 'ES']
                l2 = tmp[tmp.Language == 'L2']
                esp = tmp[tmp.Language == 'ES+']
                esm = tmp[tmp.Language == 'ES-']

                D_esp = cosine_similarity(esp[hidd_cols], esp[hidd_cols])
                D_esm = cosine_similarity(esm[hidd_cols], esm[hidd_cols])
                D_esp_es = cosine_similarity(esp[hidd_cols], l1[hidd_cols])
                D_esm_es = cosine_similarity(esm[hidd_cols], l1[hidd_cols])
                D_esp_l2 = cosine_similarity(esp[hidd_cols], l2[hidd_cols])
                D_esm_l2 = cosine_similarity(esm[hidd_cols], l2[hidd_cols])

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

                types = ['ES+', 'ES-', 'ES+ v L1', 'ES- v L1']
                mats = [D_esp, D_esm, D_esp_es, D_esm_es]

                for tp, D in zip(types, mats):
                    if grp == 'MONO' and 'L2' in tp:
                        continue
                    similarities['dataset'].append(category)
                    similarities['prob'].append(end)
                    similarities['Version'].append(grp)
                    similarities['run'].append(run)
                    similarities['Training'].append(data_label)
                    similarities['Type'].append(tp)
                    similarities['avg_dist'].append(np.triu(D, 1).mean())

                if plot:
                    if run == 0 and category == 'ESEN' and prob == 60:
                        print(
                            'Reducing the dimensionality for plotting. This will take a while.')
                        #tmp[hidd_cols] = StandardScaler().fit_transform(tmp[hidd_cols])

                        pca = PCA(n_components=50)

                        pca_res = pca.fit_transform(tmp[hidd_cols])

                        tsne = TSNE(n_components=2, perplexity=100, n_jobs=-1,
                                    random_state=args.seed)

                        tmp[['dim1', 'dim2']] = tsne.fit_transform(pca_res)

                        if data_label == 'Untrained':
                            explore_df_untrained = tmp
                        else:
                            explore_df = tmp

                        b = tmp[(tmp.Language == 'ES') |
                                (tmp.Language == 'L2')]

                        ax = sns.jointplot(x='dim1', y='dim2', kind='scatter',
                                           hue='Language', hue_order=['ES', 'L2'],
                                           palette=['C1', 'C0'], data=b, alpha=0.5,
                                           space=0.1, xlim=(-70, 70),
                                           ylim=(-70, 70), s=15)
                        ax.fig.suptitle(f"{m_name}_{data_label}")
                        ax.fig.subplots_adjust(top=0.90)
                        plt.show()

                        a = tmp[(tmp.Language == 'ES+') |
                                (tmp.Language == 'ES-')]

                        ax = sns.jointplot(x='dim1', y='dim2', kind='scatter',
                                           hue='Language', hue_order=['ES+', 'ES-'],
                                           palette=['C2', 'C4'], data=a, alpha=0.5,
                                           xlim=(-70, 70),
                                           ylim=(-70, 70),
                                           space=0.1, s=40)
                        ax.fig.suptitle(f"{m_name}_{data_label}_Exp Words")
                        ax.fig.subplots_adjust(top=0.90)
                        plt.show()

simil = pd.DataFrame(similarities)

sns.set(style='whitegrid', context='paper', palette='colorblind', font_scale=2)

g = sns.catplot(x='Type', y='avg_dist',  # hue='Version', hue_order=['MONO', 'ES-EN', 'ES-EU'],
                #palette=['C1', 'C0', 'C2'],
                palette='Greys',
                col='Training', col_order=['Untrained', 'Trained'],
                data=simil, kind='violin', inner='point')
g.axes.flatten()[0].set_ylabel('Avg. Cosine Similarity')
g.axes.flatten()[0].set_title('Untrained')
g.axes.flatten()[1].set_title('Trained')
g.axes.flatten()[0].set_xlabel('')
g.axes.flatten()[1].set_xlabel('')
plt.ylim((0, 1.))
plt.show()

#simil.to_csv('results/backup_similarities.csv', index=False, encoding='utf-8')

#%% SUPPLEMENTARY FIGURE X
sns.set(style='whitegrid', context='paper',
        palette='colorblind', font_scale=1.5)


# Similarity of letters in Embedding layer

letters = list(ascii_lowercase) + ['<s>']


d = {} 
# 'MONO': np.zeros((28, 28)),
# 'ES-EN': np.zeros((28, 28)),
# 'ES-EU': np.zeros((28, 28))

for data, category in zip(args.datafiles, args.modelfiles):
    for prob in args.probs:
        if prob == 100 and category == 'ESEU':
            continue
        
        end = f"{prob:02}-{100-prob:02}"
        m_name = f"{category}_{end}"
        
        if "100-00" in m_name:
            cat = 'MONO'
        elif "ESEN" in m_name:
            cat = 'ES-EN'
        else:
            cat = 'ES-EU'
        
        d[cat] = {}
        
        for run in range(args.n_runs):
            print(f"\n{data}: {m_name}_{run}\n")
            model = torch.load(args.model_save_file +
                               f"{m_name}/{m_name}_{run}_threshold_val_35.pt")
            model.to(args.device)
            model.eval()
            
            d[cat][run] = model.E.weight.detach().to('cpu').numpy()[:-2, :]
        
        simil = cosine_similarity(d[cat][0], d[cat][0])
        
        plt.figure(figsize=(8,6))
        g = sns.heatmap(np.tril(simil), yticklabels=letters, xticklabels=letters, 
                        cmap='vlag',
                        vmin=-1, vmax=1)
        g.set(title=cat)
        plt.yticks(rotation=0)
        plt.xticks(rotation=0)
        plt.show()
        
#%% 
sns.set(style='white', context='paper',
        palette='colorblind', font_scale=1.5)        
      
        
cats = ['ES-EN', 'ES-EU', 'MONO']

dat = pd.DataFrame()

for i,c in enumerate(cats):
    # plt.figure(figsize=(8,6))
    # plt.title(c)
    # ax = sch.dendrogram(sch.linkage(d[c], method='ward'), labels=letters,
    #                     leaf_rotation=0, leaf_font_size=12, orientation='left')
    # plt.show()
    
    # dt = d[c]
    # cluster = AgglomerativeClustering(n_clusters=5, affinity='cosine', linkage='average')
    # cluster.fit(d[c])
    tsne = PCA(n_components=2)#, perplexity=100, n_jobs=-1, random_state=args.seed-i)
    dt = tsne.fit_transform(d[c][0])
    data = pd.DataFrame({'x':dt[:, 0], 'y':dt[:, 1], 'val':letters, 'cat':c})
    dat = pd.concat([dat, data], axis=0, ignore_index=True)
    
dat['hue'] = dat['cat'].map({'ES-EN':0, 'ES-EU':2, 'MONO':1})

plt.figure(figsize=(12,10))
ax = sns.scatterplot('x', 'y', hue='cat', data=dat, palette=['C0', 'C2', 'C1'],
                     s=0, legend=True)
pal = sns.color_palette('colorblind').as_hex()
for i, point in dat.iterrows():
    ax.text(point['x'], point['y'], str(point['val']), fontsize=20, color=pal[point['hue']])  
plt.show()
    
#%%        
# Plot contrasts
from itertools import combinations

cats = ['ES-EN', 'ES-EU', 'MONO']

for cmb in combinations(cats, 2):
    sim = d[cmb[0]] - d[cmb[1]]
    
    plt.figure(figsize=(8,6))
    g = sns.heatmap(sim, yticklabels=letters,
                    cmap='vlag', vmin=-1, vmax=1)
    g.set(title=cmb)
    plt.show()
    
    
