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

random.seed(404)

path = os.getcwd()
os.chdir(path)

file_path = "RAW/"

esp = pd.read_csv(os.path.join(file_path, 'ESP.csv'), encoding='utf-8')
eng = pd.read_csv(os.path.join(file_path, 'ENG.csv'), sep=',',
                  encoding='utf-8')

eus = pd.read_csv(os.path.join(file_path, 'EUS.txt'), sep='\t')
eus.columns = ['spelling', 'cnt', 'freq_mil', 'sil', 'neighbors']
eus['len'] = eus['spelling'].apply(len)
eus['log_freq'] = eus['freq_mil'].apply(np.log)

# Read nonwords for Lexical Decision Task
nwd = pd.read_csv(os.path.join(file_path, 'LDT_NW.csv'))
nwd = nwd.dropna()
nwd.len = nwd.len.map(int)

# %% Normalizing eus data
eus = eus[(eus.log_freq >= eus.log_freq.quantile(q=0.5))]
esp = esp[(esp.zipf >= esp.zipf.quantile(q=0.5))]
eng = eng[(eng.ZipfUS >= eng.ZipfUS.quantile(q=0.5))]

nwd = nwd[(nwd.percent > 72) & (nwd.percent <= 95)]
nwd = nwd[nwd['count'] > 30]

esp = esp[(esp.len >= 3) & (esp.len <= 10)]
eng = eng[(eng.len >= 3) & (eng.len <= 10)]
eus = eus[(eus.len >= 3) & (eus.len <= 10)]

nwd = nwd[(nwd.len >= 3) & (nwd.len <= 10)]

def preprocess(st):
    st = ''.join(c for c in unicodedata.normalize('NFD', st)
                 if unicodedata.category(c) != 'Mn')
    st = re.sub(r"[^a-zA-Z]", r"", st)
    return st.lower()


esp['spelling'] = esp['spelling'].map(preprocess)
eng['spelling'] = eng['spelling'].map(preprocess)
eus['spelling'] = eus['spelling'].map(preprocess)

nwd['spelling'] = nwd['spelling'].map(preprocess)

esp = esp.drop_duplicates(subset='spelling')
eng = eng.drop_duplicates(subset='spelling')
eus = eus.drop_duplicates(subset='spelling')

nwd = nwd.drop_duplicates(subset='spelling')
nwd = nwd[~(nwd.spelling.isin(esp.spelling))]
nwd = nwd[~(nwd.spelling.isin(eng.spelling))]
nwd = nwd[~(nwd.spelling.isin(eus.spelling))]

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

nwd = nwd.sort_values(by='len')

# %% Match length distribution by sampling from the smallest values

esp_vals = esp_data.len.value_counts(
    sort=False).rename_axis('len').reset_index(name='counts')
eng_vals = eng_data.len.value_counts(
    sort=False).rename_axis('len').reset_index(name='counts')
eus_vals = eus_data.len.value_counts(
    sort=False).rename_axis('len').reset_index(name='counts')
nwd_vals = nwd.len.value_counts(sort=False).rename_axis('len').reset_index(name='counts')

new_esp = pd.DataFrame()
new_eng = pd.DataFrame()
new_eus = pd.DataFrame()
new_nwd = pd.DataFrame()

for l, es, en, eu in zip(esp_vals.len, esp_vals.counts, eng_vals.counts, eus_vals.counts):
    n = min(es, en, eu)
    repl = nwd_vals[nwd_vals.len == l]['counts'] < n
    new_nwd = pd.concat([new_nwd, nwd[nwd.len == l].sample(n, replace=repl.iloc[0])])
    new_esp = pd.concat([new_esp, esp_data[esp_data.len == l].sample(n)])
    new_eng = pd.concat([new_eng, eng_data[eng_data.len == l].sample(n)])
    new_eus = pd.concat([new_eus, eus_data[eus_data.len == l].sample(n)])

print(new_esp.len.value_counts(sort=False))
print(new_eng.len.value_counts(sort=False))
print(new_eus.len.value_counts(sort=False))
print(new_nwd.len.value_counts(sort=False))

# %% Select dev and test set sizes

n = 82

idx2 = n * 10
idx3 = n * 10
idx1 = len(esp_data) - idx2 - idx3

# Create an ldt dataframe to control for train, val, test words
ldt = new_esp
ldt['nwd'] = list(new_nwd.spelling)

esp_data = ldt.sample(frac=1.)  # Shuffle the words
eng_data = new_eng.sample(frac=1.)
eus_data = new_eus.sample(frac=1.)

splits = ['train'] * idx1 + ['val'] * idx2 + ['test'] * idx3

# %% First dataset
data = pd.DataFrame(columns=['data', 'label', 'freq', 'split'])
data['data'] = list(esp_data['data']) + list(eng_data['data'])
data['freq'] = list(esp_data['freq']) + list(eng_data['freq'])
data['label'] = ['ESP'] * len(esp_data) + ['ENG'] * len(eng_data)
data['split'] = splits*2

data['len'] = data['data'].apply(len)
print(stats.ttest_ind(data[(data.split == 'train') & (data.label == 'ESP')].len,
                      data[(data.split == 'train') & (data.label == 'ENG')].len))
print(stats.ttest_ind(data[(data.split == 'val') & (data.label == 'ESP')].len,
                      data[(data.split == 'val') & (data.label == 'ENG')].len))
print(stats.ttest_ind(data[(data.split == 'test') & (data.label == 'ESP')].len,
                      data[(data.split == 'test') & (data.label == 'ENG')].len))

data.to_csv('ESP-ENG.csv',
            index=False, encoding='utf-8')

# %% Second dataset
data = pd.DataFrame(columns=['data', 'label', 'split'])
data['data'] = list(esp_data['data']) + list(eus_data['data'])
data['freq'] = list(esp_data['freq']) + list(eus_data['freq'])
data['label'] = ['ESP'] * len(esp_data) + ['EUS'] * len(eus_data)
data['split'] = splits*2

data['len'] = data['data'].apply(len)
print(stats.ttest_ind(data[(data.split == 'train') & (data.label == 'ESP')].len,
                      data[(data.split == 'train') & (data.label == 'EUS')].len))
print(stats.ttest_ind(data[(data.split == 'val') & (data.label == 'ESP')].len,
                      data[(data.split == 'val') & (data.label == 'EUS')].len))
print(stats.ttest_ind(data[(data.split == 'test') & (data.label == 'ESP')].len,
                      data[(data.split == 'test') & (data.label == 'EUS')].len))

data.to_csv('ESP-EUS.csv',
            index=False, encoding='utf-8')

# %% Lexical Decision Dataset
LDT = pd.DataFrame(columns=['W1', 'W2', 'len'])
LDT['W1'] = list(esp_data.data)
LDT['W2'] = list(esp_data.nwd)
LDT['split'] = splits
LDT['len'] = list(new_esp.len)

LDT.to_csv('LDT.csv',
           index=False, encoding='utf-8')
