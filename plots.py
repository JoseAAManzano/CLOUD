# %% Imports
import torch
import utils
import numpy as np
import pandas as pd
import torch.nn.functional as F
import json

import seaborn as sns
sns.set(style='white', context='paper', palette='pastel', font_scale=1.5)
import matplotlib.pyplot as plt

from string import ascii_lowercase
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist

#%% Parameter variables

# def plot_letters(model_files):
#     letters = list(ascii_lowercase) + ['<s>', '</s>']
#     for m_file in model_files:
#         model = torch.load(model_file)
#         model.to('cpu')
#         model.eval()
#         tsne = TSNE(n_components=2, perplexity=100)
#         E = model.E.weight.detach().numpy()[:-1]
#         E_emb = tsne.fit_transform(E)
        
#         E_emb = pd.DataFrame(E_emb, columns=['Dim1','Dim2'])
#         E_emb['letters'] = letters
        
#         plt.figure(figsize=(7, 5))
#         ax = sns.scatterplot(x='Dim1', y='Dim2', data=E_emb)
#         sns.despine()
#         label_points(E_emb['Dim1'], E_emb['Dim2'], E_emb['letters'], ax)
#         plt.show()
    
model_file = 'models/ESEN_60-40/ESEN_60-40_0_threshold_val_35.pt'
dataset = pd.read_csv('data/ESP-ENG.csv')
vectorizer = utils.Vectorizer.from_df(dataset)
word = 'model'

model = torch.load(model_file)
model.to('cpu')
model.eval()

utils.set_all_seeds(404, 'cpu')

#%% Plot character probabilities
letters = list(ascii_lowercase) + ['<s>', '</s>']
top_k = 5
tmp = np.zeros((top_k, len(word)+1))
annot = [['' for _ in range(len(word)+1)] for _ in range(top_k)]

for i, (f_v, t_v) in vectorizer.vectorize_single_char(word):
    f_v, t_v = f_v.to('cpu'), t_v.to('cpu')

    hidden = model.init_hidden(1)

    out, out_rnn, hidden = model(f_v.unsqueeze(0), torch.LongTensor([i+1]), hidden, max_length=i+1)

    dist = torch.flatten(out[-1, :].detach())
    
    dist = dist[:-1]
    
    prb = F.softmax(dist, dim=0)    
    
    ret = prb.numpy()
    
    sorted_probs = np.sort(ret)[::-1][:top_k]
    argsort_probs = np.argsort(ret)[::-1][:top_k]
    for j in range(top_k):
        tmp[j, i] = sorted_probs[j]
        annot[j][i] = f"{letters[argsort_probs[j]].upper()}\n\n{sorted_probs[j]:.2f}"
    
plt.figure(figsize=(8,6))
sns.heatmap(data=tmp, cmap='Blues', annot=annot, fmt='',
            xticklabels=list(word.upper()) + ['</s>'],
            vmin=0., vmax=0.5)

#%% Plot word CLOUD
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
            _, out_rnn, _ = model(f_v.unsqueeze(0), torch.LongTensor([i+1]), hidden, max_length=i+1)
            hidd['Word'].append(w)
            hidd['Char'].append(i)
            hidd['Language'].append(l)
            hidd['Length'].append(len(w))
            hid = torch.flatten(out_rnn.squeeze(0)[-1].detach()).to('cpu').numpy()
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
        _, out_rnn, _ = model(f_v.unsqueeze(0), torch.LongTensor([i+1]), hidden, max_length=i+1)
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
df = pd.pivot_table(tmp, index=['Word','Length', 'Language'], values=hidd_cols,
                    aggfunc=np.mean).reset_index()
df['Language'] = df.Language.apply(lambda x: x[:-1])

sc = StandardScaler()
df[hidd_cols] = sc.fit_transform(df[hidd_cols].values)

# least = hidden_rep[hidden_rep.Length < len(word)] # Take words that are smaller
# least = least[least.Char == least.Length] # Select the last representation
# least['Char'] = len(word)

# df = hidden_rep[hidden_rep.Char == len(word)]
# df = pd.concat([df, least], axis=0)

# Compress hidden representations for plotting using PCA and TSNE
print('Reducing the dimensionality for plotting. This will take a while.')
pca = PCA(n_components=50)

pca_res = pca.fit_transform(df[hidd_cols])

tsne = TSNE(n_components=2, perplexity=100, n_jobs=-1)

df[['dim1', 'dim2']] = tsne.fit_transform(pca_res)

del pca_res, tmp

# Plot the entire dataset
ax = sns.jointplot(x='dim1', y='dim2', kind='scatter',
                   hue='Language', hue_order=['ES', 'EN'],
                   data=df, alpha=0.8, space=0.1,
                   xlim=(-70, 70), ylim=(-70, 70), s=2)
plt.show()

def print_from(df, d1_l, d1_h, d2_l, d2_h):
    return sorted(list(df[(df['dim1'] >= d1_l) & (df['dim1'] <= d1_h) &
                   (df['dim2'] >= d2_l) & (df['dim2'] <= d2_h)].Word))

list1 = print_from(df, 20, 40, 40, 60)
list2 = print_from(df, -60, -40, -40, -20)

# Compute the distance from all words to the selected word
print("Computing distances")
word_rep = df[df.Word == word][hidd_cols].values[0]

dists = []
for i, row in df.iterrows():
    if (i+1) % 1000 == 0: print("{i+1} words processed.")
    dists.append(1 - word_rep.dot(row[hidd_cols].values) / 
                 (np.linalg.norm(word_rep) * np.linalg.norm(row[hidd_cols].values))
        )

df['dist'] = dists

# Get most similar and dissimilar words
top = df.sort_values(by='dist', ascending=True)[:21]

def jitter(df, col):
    return df[col] + np.random.randn(len(df)) * (0.5 * (max(df[col]) - min(df[col])))

top['dim2_j'] = jitter(top, 'dim2')
top['dim1_j'] = jitter(top, 'dim1')

def label_points(x, y, val, ax):
    a = pd.concat({'x':x, 'y':y, 'val':val}, axis=1)
    for i, point in a.iterrows():
        if str(point['val']) == word:
            ax.text(point['x']-.02, point['y']+.02, str(point['val']), bbox=dict(ec='red', fc='w', alpha=0.7), fontsize=25)
        else:
            ax.text(point['x']-.02, point['y']+.02, str(point['val']), fontsize=20)

plt.figure(figsize=(7, 5))
ax = sns.scatterplot(x='dim1_j', y='dim2_j', hue=top.Language.tolist(), data=top,
                     hue_order=['ES', 'EN'], s=35)
sns.despine()
label_points(top['dim1_j'], top['dim2_j'], top.Word, ax)
ax.legend(loc='lower left')
plt.show()