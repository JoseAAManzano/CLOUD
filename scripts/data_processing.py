# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 15:31:40 2020

@author: josea
"""
import os
import re
import random
import unicodedata
import pandas as pd
import numpy as np

import scipy.stats as stats

random.seed(4004)

path = os.getcwd()
os.chdir(path)

file_path = "../../data/"
target_path = "../../processed_data/"

esp = pd.read_csv(os.path.join(file_path, 'ESP.csv'), encoding='utf-8')
eng = pd.read_csv(os.path.join(file_path, 'ENG2.csv'), sep=',',
                  encoding='utf-8')

eus = pd.read_csv(os.path.join(file_path, 'EUS2.txt'), sep='\t')
eus.columns = ['spelling','cnt', 'freq_mil', 'sil', 'neighbors']
eus['len'] = eus['spelling'].apply(len)
eus['log_freq'] = eus['freq_mil'].apply(np.log)

# %% Normalizing eus data
eus = eus[(eus.log_freq >= eus.log_freq.quantile(q=0.5))]

esp = esp[(esp.zipf >= esp.zipf.quantile(q=0.5))]
eng = eng[(eng.ZipfUS >= eng.ZipfUS.quantile(q=0.5))]

esp = esp[(esp.len >= 3) & (esp.len <= 10)]
eng = eng[(eng.len >= 3) & (eng.len <= 10)]
eus = eus[(eus.len >= 3) & (eus.len <= 10)]

def preprocess(st):
    st = ''.join(c for c in unicodedata.normalize('NFD', st)
                 if unicodedata.category(c) != 'Mn')
    st = re.sub(r"[^a-zA-Z]", r"", st)
    return st.lower()

esp['spelling'] = esp['spelling'].map(preprocess)
eng['spelling'] = eng['spelling'].map(preprocess)
eus['spelling'] = eus['spelling'].map(preprocess)

esp = esp.drop_duplicates(subset='spelling')
eng = eng.drop_duplicates(subset='spelling')
eus = eus.drop_duplicates(subset='spelling')

cols = ['data', 'freq']

esp_data = esp[['spelling', 'zipf']]
esp_data.columns = cols
eng_data = eng[['spelling', 'ZipfUS']]
eng_data.columns = cols
eus_data = eus[['spelling', 'freq_mil']]
eus_data.columns = cols

esp_data['len'] = esp_data.data.apply(len)
eng_data['len'] = eng_data.data.apply(len)
eus_data['len'] = eus_data.data.apply(len)

esp_data = esp_data.sort_values(by='len')
eng_data = eng_data.sort_values(by='len')
eus_data = eus_data.sort_values(by='len')

# %% Match length distribution by sampling from the smallest values

esp_vals = esp_data.len.value_counts(sort=False).rename_axis('len').reset_index(name='counts')
eng_vals = eng_data.len.value_counts(sort=False).rename_axis('len').reset_index(name='counts')
eus_vals = eus_data.len.value_counts(sort=False).rename_axis('len').reset_index(name='counts')

new_esp = pd.DataFrame()
new_eng = pd.DataFrame()
new_eus = pd.DataFrame()

for l, es, en, eu in zip(esp_vals.len, esp_vals.counts, eng_vals.counts, eus_vals.counts):
    n = min(es, en, eu)
    new_esp = pd.concat([new_esp, esp_data[esp_data.len == l].sample(n)])
    new_eng = pd.concat([new_eng, eng_data[eng_data.len == l].sample(n)])
    new_eus = pd.concat([new_eus, eus_data[eus_data.len == l].sample(n)])

esp_data = new_esp.sample(len(new_esp) - 6) # Making the size a round number
eng_data = new_eng.sample(len(new_esp) - 6)
eus_data = new_eus.sample(len(new_esp) - 6)

print(esp_data.len.value_counts(sort=False))
print(eng_data.len.value_counts(sort=False))
print(eus_data.len.value_counts(sort=False))

# %% Select dev and test set sizes

n = 128

idx2 = n * 10
idx3 = n * 10
idx1 = len(esp_data) - idx2 - idx3

esp_data = esp_data.sample(frac=1.) # Shuffle again
eng_data = eng_data.sample(frac=1.)
eus_data = eus_data.sample(frac=1.)

# %% First dataset
data = pd.DataFrame(columns=['data', 'label', 'freq', 'split'])
data['data'] = list(esp_data['data']) + list(eng_data['data'])
data['freq'] = list(esp_data['freq']) + list(eng_data['freq'])
data['label'] = ['ESP'] * len(esp_data) + ['ENG'] * len(eng_data)
splits = ['train']*idx1 + ['val']*idx2 + ['test'] * \
    idx3 + ['train']*idx1 + ['val']*idx2 + ['test']*idx3
data['split'] = splits

data['len'] = data['data'].apply(len)
print(stats.ttest_ind(data[(data.split=='train') & (data.label=='ESP')].len,
                      data[(data.split=='train') & (data.label=='ENG')].len))
print(stats.ttest_ind(data[(data.split=='val') & (data.label=='ESP')].len,
                      data[(data.split=='val') & (data.label=='ENG')].len))
print(stats.ttest_ind(data[(data.split=='test') & (data.label=='ESP')].len,
                      data[(data.split=='test') & (data.label=='ENG')].len))

data.to_csv(os.path.join(target_path, 'ESP-ENG.csv'),
            index=False, encoding='utf-8')

# %% Second dataset
data = pd.DataFrame(columns=['data', 'label', 'split'])
data['data'] = list(esp_data['data']) + list(eus_data['data'])
data['freq'] = list(esp_data['freq']) + list(eus_data['freq'])
data['label'] = ['ESP'] * len(esp_data) + ['EUS'] * len(eus_data)
splits = ['train']*idx1 + ['val']*idx2 + ['test'] * \
    idx3 + ['train']*idx1 + ['val']*idx2 + ['test']*idx3
data['split'] = splits

data['len'] = data['data'].apply(len)
print(stats.ttest_ind(data[(data.split=='train') & (data.label=='ESP')].len,
                      data[(data.split=='train') & (data.label=='EUS')].len))
print(stats.ttest_ind(data[(data.split=='val') & (data.label=='ESP')].len,
                      data[(data.split=='val') & (data.label=='EUS')].len))
print(stats.ttest_ind(data[(data.split=='test') & (data.label=='ESP')].len,
                      data[(data.split=='test') & (data.label=='EUS')].len))

data.to_csv(os.path.join(target_path, 'ESP-EUS.csv'),
            index=False, encoding='utf-8')
