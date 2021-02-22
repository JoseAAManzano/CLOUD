# %%
import torch
import pandas as pd
import utils
import numpy as np
import torch.nn.functional as F
from string import ascii_lowercase
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='paper', style='white', palette='pastel')

from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist

def plot_word_probabilities(word, model_file, vectorizer, top_k=6):
    letters = list(ascii_lowercase) + ['<s>', '</s>']
    
    model = torch.load(model_file)
    model.to('cpu')
    model.eval()
    
    tmp = np.zeros((top_k, len(word)+1))
    annot = [['' for _ in range(len(word)+1)] for _ in range(top_k)]
    
    for i, (f_v, t_v) in vectorizer.vectorize_single_char(word):
        f_v, t_v = f_v.to('cpu'), t_v.to('cpu')
    
        hidden = model.init_hidden(1)
    
        out, hidden = model(f_v.unsqueeze(0), torch.LongTensor([i+1]), hidden, max_length=i+1)
    
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
    sns.heatmap(data=tmp, cmap='Blues', annot=annot, fmt='', xticklabels=list(word.upper()) + ['</s>'],
                vmin=0., vmax=0.5)


def plot_word_cloud(word, model_file, dataset, vectorizer, dist='cosine', n=15):
    model = torch.load(model_file)
    model.to('cpu')
    model.eval()
    
    hidd_cols = [f"hid_{i+1}" for i in range(model.rnn.hidden_size)]
        
    # Get hidden representation for the word using the model
    w_v, _, w_len = vectorizer.vectorize(word).values()
    w_len = torch.LongTensor([w_len])
    
    hidden = model.init_hidden(1)
    
    _, hidden = model(w_v.unsqueeze(0), w_len, hidden)
    
    word_rep = torch.flatten(hidden[0].detach()).numpy()
    
    hidden_file = model_file.split('/')[-1].split('_t')[0]
        
    df = pd.DataFrame()
    for dt in ['train', 'val', 'test']:    
        tmp = pd.read_json(f"hidden/{dt}_hidden_{hidden_file}.json",
                       encoding='utf-8')
        
        tmp = tmp[tmp.char == w_len.item()-1]
        
        df = pd.concat([df, tmp], axis=0, ignore_index=True)
    
    del tmp
    
    # Compress hidden representations for plotting using PCA and TSNE
    pca = PCA(n_components=50)
    
    pca_res = pca.fit_transform(df[hidd_cols])
    
    tsne = TSNE(n_components=2, perplexity=100, n_jobs=-1)
    
    df[['dim1', 'dim2']] = tsne.fit_transform(pca_res)
    
    df['Language'] = df['lang']
    dt = df.sample(frac=.2)
    ax = sns.jointplot(x='dim1', y='dim2', hue='Language', hue_order=['ES', 'EN'], data=dt)
    plt.show()
    
    # Compute the distance from all words to the selected word
    dists = []
    for i, row in df.iterrows():
        dists.append(pdist([word_rep, row[hidd_cols].values], dist)[0])
    
    df['dist'] = dists
    
    df['lang'] = df['label'].map(lambda x: x[:-1])
    
    # Get most similar and dissimilar words
    top = df.sort_values(by='dist', ascending=True)[:n+1]
    
    def jitter(df, col):
        return df[col] + np.random.randn(len(df)) * (0.5 * (max(df[col]) - min(df[col])))

    top['dim2_j'] = jitter(top, 'dim2')
    top['dim1_j'] = jitter(top, 'dim1')
    
    plt.figure(figsize=(7, 5))
    ax = sns.scatterplot(x='dim1_j', y='dim2_j', hue=top.lang.tolist(), data=top,  hue_order=['ES', 'EN'])
    sns.despine()
    label_points(top['dim1_j'], top['dim2_j'], top.word, ax)
    ax.legend(loc='upper left')
    plt.show()
    
def print_from(df, d1_l, d1_h, d2_l, d2_h):
    return list(df[(df['dim1'] >= d1_l) & (df['dim1'] <= d1_h) &
                   (df['dim2'] >= d2_l) & (df['dim2'] <= d2_h)].word)
    
def label_points(x, y, val, ax):
    a = pd.concat({'x':x, 'y':y, 'val':val}, axis=1)
    for i, point in a.iterrows():
        if str(point['val']) == word:
            ax.text(point['x']-.002, point['y']+.002, str(point['val']), bbox=dict(ec='red', fc='w', alpha=0.7), fontsize=18)
        else:
            ax.text(point['x']-.002, point['y']+.002, str(point['val']), fontsize=16)

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
    
if __name__ == '__main__':
    model_file = 'models/LSTM/128/ESEN_60-40/ESEN_60-40_0_threshold_val_34.pt'
    dataset = pd.read_csv('data/ESP-ENG.csv')
    vectorizer = utils.Vectorizer.from_df(dataset)
    word = 'model'
    
    plot_word_probabilities(word, model_file, vectorizer, top_k=5)
    
    plot_word_cloud(word, model_file, dataset, vectorizer, n=15, dist='cosine')
    
