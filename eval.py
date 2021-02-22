# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 13:48:18 2020

@author: josea
"""

# %% Imports
from scipy.spatial import distance
import utils
import torch
import math
import torch.nn.functional as F
import seaborn as sns
import pandas as pd
import collections
import torch.nn as nn
import numpy as np

from argparse import Namespace
from collections import defaultdict

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import metrics as mtr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

utils.set_all_seeds(args.seed, args.device)
# %% Helper


def get_acc(model, vectorizer, data, device):
    model.to(device)
    acc = 0.0
    model.eval()
    for i, word in enumerate(data):
        from_v, to_v = vectorizer.vectorize(word)
        from_v, to_v = from_v.to(device), to_v.to(device)
        from_v = from_v.unsqueeze(0)

        hidden = model.initHidden(1, device)

        out, _ = model(from_v, hidden)

        char_acc = utils.compute_accuracy(out, to_v,
                                          vectorizer.data_vocab.PAD_idx)
        acc += (char_acc - acc) / (i + 1)

    return acc


def get_distribution_from_context(model, context, vectorizer):
    hidden = model.initHidden(1, args.device)

    for i, (f_v, t_v) in vectorizer.vectorize_single_char(context):
        f_v = f_v.to(args.device)
        out, hidden = model(f_v.unsqueeze(1), hidden)
    dist = torch.flatten(out.detach()).to('cpu')
    # Take only valid continuations (letters + SOS)
    dist = dist[:-2] + dist[-1]
    return F.softmax(dist, dim=0).numpy()


def calculate_divergence(words, trie, model, vectorizer=None):
    running_div = 0.0
    for i, word in enumerate(words):
        if isinstance(model, utils.CharNGram):
            m_dist = np.float32(
                list(model.get_distribution_from_context(word[:-1]).values()))
        else:
            m_dist = get_distribution_from_context(
                model, word[:-1], vectorizer)

        e_dist = np.float32(trie.get_distribution_from_context(word[:-1]))

        pos = (m_dist != 0.) & (e_dist != 0.)

        m_dist = m_dist[pos]
        e_dist = e_dist[pos]

        KL = np.sum(e_dist * (np.log2(e_dist) - np.log2(m_dist)))

        running_div = (KL - running_div) / (i + 1)
    return running_div

# def get_hidden_representation(model, vectorizer, df, device):
#     ret = pd.DataFrame()
#     words = df.data
#     model.eval()
#     for i, word in enumerate(words):
#         from_v, to_v = vectorizer.vectorize(word)
#         from_v, to_v = from_v.to(device), to_v.to(device)
#         from_v = from_v.unsqueeze(0)

#         hidden = model.initHidden(1, device)

#         out, hidden = model(from_v, hidden)

#         ret[word] = torch.flatten(hidden[0].detach()).to('cpu').numpy()
#     ret = ret.T
#     ret['data'] = ret.index

#     return df.merge(ret, on='data')


# %% Set-up experiments
tmp = defaultdict(list)
# results_similarity = pd.DataFrame()

eval_words = pd.read_csv(args.csv + 'exp_words.csv')
es0_words = list(eval_words[eval_words.label == 'ES-'].data)
es1_words = list(eval_words[eval_words.label == 'ES+'].data)

for data, category in zip(args.datafiles, args.modelfiles):
    for prob in [50, 99]:
        end = f"{prob:02}-{100-prob:02}"
        m_name = f"{category}{end}"

        dataset = utils.TextDataset.load_dataset_and_make_vectorizer(
            args.csv + data,
            p=prob / 100, seed=args.seed)
        vectorizer = dataset.get_vectorizer()

        train_words = list(dataset.train_df.data)
        # trie_train = utils.Trie()
        # trie_train.insert_many([list(w) + ['</s>'] for w in train_words])

        test_words = list(dataset.test_df.data)
        # trie_test = utils.Trie()
        # trie_test.insert_many([list(w) + ['</s>'] for w in test_words])

        for run in range(args.n_runs):
            print(f"\n{data}: {m_name}_{run}\n")

            # Get N-gram models with different laplace constant
            ngrams = {}
            for n in range(2, 5):
                ngrams[f"{n}-gram"] = utils.CharNGram(train_words, n,
                                                      laplace=(run + 1) * 0.2)

            for name, m in ngrams.items():
                tmp["model"].append(name)
                tmp["prob"].append(end)
                tmp["data"].append(category[:-1])
                tmp["run"].append(run)
                tmp["accuracy_train"].append(m.calculate_accuracy(train_words))
                tmp["perplexity_train"].append(m.perplexity(train_words))
                tmp["accuracy"].append(m.calculate_accuracy(test_words))
                tmp["perplexity"].append(m.perplexity(test_words))
                #tmp["KL_train"].append(calculate_divergence(train_words, trie_train, m))
                #tmp["KL_test"].append(calculate_divergence(test_words, trie_test, m))
                tmp['ES+'].append(m.calculate_accuracy(es1_words))
                tmp['ES-'].append(m.calculate_accuracy(es0_words))

            lstm_model = torch.load(args.model_save_file +
                                    f"{m_name}/{m_name}_{run}.pt")
            lstm_model.eval()
            lstm_model.to(args.device)

            dataset.set_split('train')
            batch_generator = utils.generate_batches(
                dataset, batch_size=args.batch_size, device=args.device)

            acc = 0.0
            perp = 0.0

            for batch_id, batch_dict in enumerate(batch_generator):
                hidden = lstm_model.initHidden(args.batch_size, args.device)

                out, hidden = lstm_model(batch_dict['X'], hidden)

                acc_chars = utils.compute_accuracy(
                    out, batch_dict['Y'], vectorizer.data_vocab.PAD_idx)
                perp_chars = F.cross_entropy(*utils.normalize_sizes(
                    out,
                    batch_dict['Y']),
                    ignore_index=vectorizer.data_vocab.PAD_idx
                )
                acc += (acc_chars - acc) / (batch_id + 1)
                perp += (perp_chars.item() - perp) / (batch_id + 1)

            tmp["model"].append("LSTM")
            tmp["prob"].append(end)
            tmp["data"].append(category[:-1])
            tmp['run'].append(run)
            tmp["accuracy_train"].append(acc)
            tmp["perplexity_train"].append(math.exp(perp))

            dataset.set_split('test')
            batch_generator = utils.generate_batches(
                dataset, batch_size=args.batch_size, device=args.device)

            acc = 0.0
            perp = 0.0

            for batch_id, batch_dict in enumerate(batch_generator):
                hidden = lstm_model.initHidden(args.batch_size, args.device)

                out, hidden = lstm_model(batch_dict['X'], hidden)

                acc_chars = utils.compute_accuracy(
                    out, batch_dict['Y'], vectorizer.data_vocab.PAD_idx)
                perp_chars = F.cross_entropy(*utils.normalize_sizes(
                    out,
                    batch_dict['Y']),
                    ignore_index=vectorizer.data_vocab.PAD_idx
                )
                acc += (acc_chars - acc) / (batch_id + 1)
                perp += (perp_chars.item() - perp) / (batch_id + 1)

            tmp["accuracy"].append(acc)
            tmp["perplexity"].append(math.exp(perp))
            # tmp["KL_train"].append(calculate_divergence(train_words, trie_train,
            #                                             lstm_model, vectorizer))
            # tmp["KL_test"].append(calculate_divergence(test_words, trie_test,
            # lstm_model, vectorizer))

            tmp["ES-"].append(get_acc(lstm_model, vectorizer,
                                      es0_words, args.device))
            tmp["ES+"].append(get_acc(lstm_model, vectorizer,
                                      es1_words, args.device))

results = pd.DataFrame(tmp)

# %% Plots
exp_data = results[['model', 'prob', 'data', 'run', 'ES-', 'ES+']]
exp_data = pd.melt(exp_data, id_vars=['model', 'prob', 'data', 'run'],
                   value_vars=['ES-', 'ES+'])

results_data = results[['model',
                        'prob',
                        'data',
                        'run',
                        'accuracy_train',
                        'perplexity_train',
                        'accuracy',
                        'perplexity']]
# 'KL_train','KL_test']]

results_acc = pd.melt(
    results_data,
    id_vars=[
        'model',
        'prob',
        'data',
        'run'],
    value_vars=[
        'accuracy_train',
        'accuracy'],
    var_name='split',
    value_name='acc')
results_acc['split'] = np.where(
    results_acc.split == 'accuracy_train', 'train', 'test')

g = sns.catplot(x="model", y="acc", hue="split", hue_order=['train', 'test'],
                row="data", col='prob', kind='bar',
                data=results_acc, palette="Greens")
g.set(ylim=(0, 50))

results_perp = pd.melt(
    results_data,
    id_vars=[
        'model',
        'prob',
        'data',
        'run'],
    value_vars=[
        'perplexity_train',
        'perplexity'],
    var_name='split',
    value_name='perp')
results_perp['split'] = np.where(
    results_perp.split == 'perplexity_train', 'train', 'test')

g = sns.catplot(x="model", y="perp", hue="split", hue_order=['train', 'test'],
                row="data", col='prob', kind='bar',
                data=results_perp, palette="Greens")
g.set(ylim=(0, 15))

sns.catplot(x="prob", y="value", hue="variable", row="data", col="model",
            data=exp_data, kind="bar", palette="Greens")

# results_kl = pd.melt(results_data, id_vars=['model', 'prob', 'data', 'run'],
#                       value_vars=['KL_train', 'KL_test'], var_name='split',
#                       value_name='KL')
# results_kl['split'] = np.where(results_kl.split == 'KL_train', 'train', 'test')
# results_kl['model_split'] = results_kl.model + "_" + results_kl.split
# results_kl= results_kl.sort_values(by=['model_split', 'prob'])

# g = sns.catplot(x="prob", y="KL", hue="model", row="data", col='split', kind='point',
# data=results_kl, palette="Reds", ci='sd', linestyles=['-', '--']*5)

# # %% Distribution of last character
# # TODO Compare distribution of last character in test words
# # Create a list of distributions for each word [len(test_words), len(vocab)]
# # Average the KL divergence/other distance metric over all the words in test
# # Plot

# for data, category in zip(args.datafiles, args.modelfiles):
#     for prob in args.probs:
#         end = f"{prob:02}-{100-prob:02}"
#         m_name = f"{category}{end}"

#         dataset = utils.TextDataset.load_dataset_and_make_vectorizer(
#             args.csv + data,
#             p=prob / 100, seed=args.seed)
#         vectorizer = dataset.get_vectorizer()

#         train_words = list(dataset.train_df.data)
#         trie_train = utils.Trie().insert_many(
#             [list(w) + '</s>' for w in train_words])

#         test_words = list(dataset.test_df.data)
#         trie_test = utils.Trie().insert_many(
#             [list(w) + '</s>' for w in test_words])

#         tmp = collections.defaultdict(list)

#         for run in range(args.n_runs):
#             print(f"\n{data}: {m_name}_{run}\n")

#             # Get N-gram models with different laplace constant
#             ngrams = {}
#             for n in range(2, 5):
#                 ngrams[f"{n}-gram"] = utils.CharNGram(train_words, n,
#                                                       laplace=(run + 1) * 0.2)

#             for name, m in ngrams.items():
#                 tmp["model"].append(name)
#                 tmp["prob"].append(end)
#                 tmp["data"].append(category[:-1])
#                 tmp["run"].append(run)
#                 tmp["accuracy_train"].append(m.calculate_accuracy(train_words))
#                 tmp["perplexity_train"].append(m.perplexity(train_words))
#                 tmp["accuracy"].append(m.calculate_accuracy(test_words))
#                 tmp["perplexity"].append(m.perplexity(test_words))
#                 tmp["KL_train"].append(
#                     calculate_divergence(
#                         train_words, trie_train, m))
#                 tmp["KL_test"].append(
#                     calculate_divergence(
#                         test_words, trie_test, m))


# %% Hidden representation for each time-point for each word
# First is necessary to run the readout.py file to produce the representations
hidd_cols = [f"hid_{str(i+1)}" for i in range(args.hidden_dims)]
#c_cols = [f"cel_{i}" for i in hidd_cols]

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
    
    res = defaultdict(float)
    for met, func in metrics.items():
        if met == 'ROC_AUC':
            res[met] = func(y_test, preds, average='weighted')
        else:
            res[met] = func(y_test, preds)
    return res

for data, category in zip(args.datafiles, args.modelfiles):
    for prob in args.probs:
        end = f"{prob:02}-{100-prob:02}"
        m_name = f"{category}_{end}"
        
        test_dataset = pd.DataFrame()
        val_dataset = pd.DataFrame()
        for run in range(args.n_runs):
            hdn = pd.read_json(f"hidden/val_hidden_{m_name}_{run}.json",encoding='utf-8')
            # cll = pd.read_json(f"{args.save_file}/val_cell_{m_name}_{run}.json",
            #                    encoding='utf-8')
            # cll.columns = list(cll.columns)[:-128] + c_cols
            # tmp = pd.merge(hdn, cll, how='left',
            #                    left_on=list(hdn.columns)[:-128],
            #                    right_on=list(cll.columns)[:-128])
            
            val_dataset = pd.concat([val_dataset, hdn], axis=0, ignore_index=True)
            
            hdn = pd.read_json(f"hidden/test_hidden_{m_name}_{run}.json",encoding='utf-8')
            # cll = pd.read_json(f"{args.save_file}/test_cell_{m_name}_{run}.json",
            #                    encoding='utf-8')
            # cll.columns = list(cll.columns)[:-128] + c_cols
            # tmp = pd.merge(hdn, cll, how='left',
            #                    left_on=list(hdn.columns)[:-128],
            #                    right_on=list(cll.columns)[:-128])
            
            test_dataset = pd.concat([test_dataset, hdn], axis=0, ignore_index=True)
            del hdn
            
        for run in range(5):
            train_data = val_dataset[val_dataset.run == run]
            test_data = test_dataset[test_dataset.run == run]
            for T in range(11):
                model_data = train_data[train_data.char == T]
                model_test = test_data[test_data.char == T]

                X_train1 = model_data[hidd_cols].values
                # X_train2 = model_data[c_cols].values
                y_train = model_data.label.values
                                
                X_test1 = model_test[hidd_cols].values
                # X_test2 = model_test[c_cols].values
                y_test = model_test.label.values
                
                m1 = LogisticRegression(max_iter=10e9, random_state=args.seed)
                
                res_x1 = pred_scores(X_train1, X_test1, y_train, y_test, m1,
                                     metrics)
                # res_x2 = pred_scores(X_train2, X_test2, y_train, y_test, m1,
                #                      metrics)

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
                    res['hid_' + met].append(val)
                
                # for met, val in res_x2.items():
                #     res['cel_' + met].append(val)

                print(
                    f"{m_name}_{run} char-{T}: {res['hid_Accuracy'][-1]:.2f}")
                
res = pd.DataFrame(res)

res.to_csv('results/backup_readout_prediction.csv', index=False, encoding='utf-8')       

# %% Plots
import scipy.stats as stats

readout = pd.read_csv('results/backup_readout_prediction.csv', encoding='utf-8')

# readout = readout[(readout.char > 0) & (readout.char < 11)]

readout.char = readout.char + 1

pivot_esen = readout[readout.dataset=='ESEN'].pivot_table(index=['run', 'char'],
                              columns='group',
                              values='hid_ROC_AUC').reset_index()
for ch in range(1, 12):
    tmp = pivot_esen[pivot_esen.char == ch]
    _, pval = stats.ttest_ind(tmp['ES-EN'], tmp['MONO'])
    st = '*' if pval < 0.05/11 else 'n.s.'
    print(f"ES-EN-{ch}: {st}")

pivot_eseu = readout[readout.dataset=='ESEU'].pivot_table(index=['run', 'char'],
                              columns='group',
                              values='hid_ROC_AUC').reset_index()
for ch in range(1, 12):
    tmp = pivot_eseu[pivot_eseu.char == ch]
    _, pval = stats.ttest_ind(tmp['ES-EU'], tmp['MONO'])
    st = '*' if pval < 0.05/11 else 'n.s.'
    print(f"ES-EU-{ch}: {st}")


g = sns.catplot(x='char', y='hid_ROC_AUC', hue='group', hue_order=['MONO', 'ES-EN', 'ES-EU'],
                col='dataset',
                data=readout, kind='point', palette='pastel', ci=99)
g.set(ylim=(0.49, 1.))
g.axes.flatten()[0].fill([2.5, 9.5, 9.5, 2.5], [0.4,0.4,1,1], 'k', alpha=0.2)
g.axes.flatten()[0].set_ylabel('AUROC', fontsize=15)
g.axes.flatten()[1].fill([2.5, 9.5, 9.5, 2.5], [0.4,0.4,1,1], 'k', alpha=0.2)

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

# %% Clustering of hidden representations

hidd_cols = [str(i) for i in range(args.hidden_dim)]

for data, category in zip(args.datafiles, args.modelfiles):
    for prob in [50, 100]:
        end = f"{prob:02}-{100-prob:02}"
        m_name = f"{category}{end}"

        dataset = pd.DataFrame()
        for run in range(args.n_runs):
            tmp = pd.read_json(f"{args.save_file}/{m_name}_{run}.json",
                               encoding='utf-8')
            dataset = pd.concat([dataset, tmp], axis=0, ignore_index=True)

        dataset['len'] = dataset.word.map(len)

        rdm_data = dataset[(dataset.char == 5) & (dataset.run == 1)]
        rdm_data = rdm_data.sort_values(by='label')
        ticks = list(rdm_data.label)

        rdm_data = np.array(rdm_data[hidd_cols].values)
        rdm_data = (rdm_data - rdm_data.mean()) / rdm_data.std()

        RDM = distance.squareform(distance.pdist(rdm_data, 'euclidean'))
        RDM = (RDM - RDM.min()) / (RDM.max() - RDM.min())

        plt.figure()
        plt.imshow(RDM, cmap='coolwarm')
        plt.axis('off')
        plt.title(f"Standardized euclidean distance {m_name}_{0}_5")
        plt.colorbar()
        plt.show()

# Can just use the previous dataframe and take the last character for each word
# Or get RDMs for each character

#         tsne = TSNE(n_components=2, n_jobs=-1, random_state=args.seed)

#         repr_df = get_hidden_representation(lstm_model, vectorizer,
#                                             test_df, args.device)

#         repr_df[['tsne1','tsne2']] = tsne.fit_transform(repr_df.iloc[:, 3:])
#         repr_df['model'] = end
#         repr_df['dataset'] = data

#         hidden_repr = pd.concat([hidden_repr, repr_df], ignore_index=True)

#         eval_df = get_hidden_representation(lstm_model, vectorizer,
#                                             eval_words, args.device)

#         eval_df[['tsne1','tsne2']] = tsne.fit_transform(eval_df.iloc[:, 2:])
#         eval_df['model'] = end
#         eval_df['dataset'] = data

#         eval_repr = pd.concat([eval_repr, eval_df], ignore_index=True)

# g = sns.FacetGrid(hidden_repr, col="model", row="dataset", hue="label")
# g.map(sns.scatterplot, "tsne1", "tsne2", alpha=0.7)
# g.add_legend()

# g = sns.FacetGrid(eval_repr, col="model", row="dataset", hue="label")
# g.map(sns.scatterplot, "tsne1", "tsne2", alpha=0.7)
# g.add_legend()

# %% Use test words to mimic learning during the blocks
eval_words = pd.read_csv(args.csv + 'exp_words.csv')

reps_per_block = 2
familiarization = False
lr = 0.001

esm = eval_words[eval_words.label == 'ES-']

utils.set_all_seeds(args.seed, args.device)

res = defaultdict(list)
for data, category in zip(args.datafiles, args.modelfiles):
    for prob in [50, 100]:
        if prob == 100 and category[:-1] == 'ESEU':
            continue
        end = f"{prob:02}-{100-prob:02}"
        m_name = f"{category}{end}"

        dataset = utils.TextDataset.load_dataset_and_make_vectorizer(
            args.csv + data,
            p=prob / 100, seed=args.seed)
        vectorizer = dataset.get_vectorizer()

        for run in range(args.n_runs):
            print(f"\n{data}: {m_name}_{run}\n")
            lstm_model = torch.load(args.model_save_file +
                                    f"{m_name}/{m_name}_{run}.pt")

            lstm_model.train()
            lstm_model.to(args.device)
            loss_fn = nn.CrossEntropyLoss(
                ignore_index=vectorizer.data_vocab.PAD_idx)
            optim = torch.optim.RMSprop(
                lstm_model.parameters(),
                lr=lr)
            
            if familiarization:
                for _ in range(1):
                    esm = esm.sample(frac=1)
                    
                    for word, lab in zip(esm.data, esm.label):
                        optim.zero_grad()
                        f_v, t_v = vectorizer.vectorize(word)
                        f_v, t_v = f_v.to(args.device), t_v.to(args.device)
                        hidden = lstm_model.initHidden(1, args.device)
                        out, _ = lstm_model(f_v.unsqueeze_(0), hidden)
        
                        loss = loss_fn(*utils.normalize_sizes(out, t_v))
                        loss.backward()
                        optim.step()
            
            for it in range(5):
                for tr in range(reps_per_block):
                    esm = esm.sample(frac=1)
                    for word, lab in zip(esm.data, esm.label):
                        optim.zero_grad()
                        f_v, t_v = vectorizer.vectorize(word)
                        f_v, t_v = f_v.to(args.device), t_v.to(args.device)
                        hidden = lstm_model.initHidden(1, args.device)
                        out, _ = lstm_model(f_v.unsqueeze_(0), hidden)
    
                        loss = loss_fn(*utils.normalize_sizes(out, t_v))
                        loss.backward()
                        optim.step()
    
                        acc = utils.compute_accuracy(out, t_v,
                                                     vectorizer.data_vocab.PAD_idx)
    
                        if prob == 50:
                            if category[:-1] == 'ESEN':
                                grp = 'ES-EN'
                            else:
                                grp = 'ES-EU'
                        else:
                            grp = 'MONO'
    
                        res['dataset'].append(category[:-1])
                        res['prob'].append(end)
                        res['run'].append(run)
                        res['word'].append(word)
                        res['Group'].append(grp)
                        res['label'].append(lab)
                        res['block'].append(it + 1)
                        res['trl'].append(tr+1)
                        res['acc'].append(acc)
                        res['loss'].append(loss.item())


esp = eval_words[eval_words.label == 'ES+']

for data, category in zip(args.datafiles, args.modelfiles):
    for prob in [50, 100]:
        if prob == 100 and category[:-1] == 'ESEU':
            continue
        end = f"{prob:02}-{100-prob:02}"
        m_name = f"{category}{end}"

        dataset = utils.TextDataset.load_dataset_and_make_vectorizer(
            args.csv + data,
            p=prob / 100, seed=args.seed)
        vectorizer = dataset.get_vectorizer()

        for run in range(args.n_runs):
            print(f"\n{data}: {m_name}_{run}\n")
            lstm_model = torch.load(args.model_save_file +
                                    f"{m_name}/{m_name}_{run}.pt")

            lstm_model.train()
            lstm_model.to(args.device)
            loss_fn = nn.CrossEntropyLoss(
                ignore_index=vectorizer.data_vocab.PAD_idx)
            optim = torch.optim.RMSprop(
                lstm_model.parameters(),
                lr=lr)
            
            if familiarization:
                for _ in range(1):
                    esp = esp.sample(frac=1)
                    
                    for word, lab in zip(esp.data, esp.label):
                        optim.zero_grad()
                        f_v, t_v = vectorizer.vectorize(word)
                        f_v, t_v = f_v.to(args.device), t_v.to(args.device)
                        hidden = lstm_model.initHidden(1, args.device)
                        out, _ = lstm_model(f_v.unsqueeze_(0), hidden)
        
                        loss = loss_fn(*utils.normalize_sizes(out, t_v))
                        loss.backward()
                        optim.step()
            
            
            for it in range(5):
                for tr in range(reps_per_block):
                    esp = esp.sample(frac=1)
                    for word, lab in zip(esp.data, esp.label):
                        optim.zero_grad()
                        f_v, t_v = vectorizer.vectorize(word)
                        f_v, t_v = f_v.to(args.device), t_v.to(args.device)
                        hidden = lstm_model.initHidden(1, args.device)
                        out, _ = lstm_model(f_v.unsqueeze_(0), hidden)
    
                        loss = loss_fn(*utils.normalize_sizes(out, t_v))
                        loss.backward()
                        optim.step()
    
                        acc = utils.compute_accuracy(out, t_v,
                                                     vectorizer.data_vocab.PAD_idx)
    
                        if prob == 50:
                            if category[:-1] == 'ESEN':
                                grp = 'ES-EN'
                            else:
                                grp = 'ES-EU'
                        else:
                            grp = 'MONO'
    
                        res['dataset'].append(category[:-1])
                        res['prob'].append(end)
                        res['run'].append(run)
                        res['word'].append(word)
                        res['Group'].append(grp)
                        res['label'].append(lab)
                        res['block'].append(it + 1)
                        res['trl'].append(tr+1)
                        res['acc'].append(acc)
                        res['loss'].append(loss.item())

res = pd.DataFrame(res)

sns.set(style='whitegrid')

g = sns.catplot(x='block', y='acc', hue='Group', hue_order=['MONO','ES-EN','ES-EU'],
                col='label', col_order=['ES-', 'ES+'],
                data=res, kind='point', ci=None)
g.set(ylim=(0, 100))

res.to_csv('simulation_results.csv', index=False, encoding='utf-8')