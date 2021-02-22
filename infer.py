# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 19:50:49 2020

@author: josea
"""
import torch
import helpers
import string

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

all_letters = string.ascii_lowercase


stoi = {'<SOS>': 0, '<EOS>': 1}
for l in all_letters:
    stoi[l] = len(stoi)

itos = {i: c for c, i in stoi.items()}

n_letters = len(stoi)

model = torch.load('models/bil-50_50.pt')


def sample(language, max_len=15, temp=0.8):
    with torch.no_grad():
        cat_ = torch.tensor([1., 0.] if language ==
                            'ESP' else [0., 1.]).view(1, -1)
        in_ = helpers.line2tensor([0], n_letters)
        hidden = model.initHidden(device)

        cat_ = cat_.to(device)

        output_word = ''

        for _ in range(max_len):
            in_ = in_.to(device)
            output, hidden = model(in_[0], cat_, hidden)
            output_dist = output.data.view(-1).div(temp).exp()
            topi = torch.multinomial(output_dist, 1)[0]
            # _, topi = output.topk(1)
            if topi.item() == stoi['<EOS>']:
                break
            else:
                letter = itos[topi.item()]
                output_word += letter
            in_ = helpers.line2tensor([stoi[letter]], n_letters)
        return output_word


for i in range(100):
    if i % 2 == 0:
        print(f"ESP: {sample('ESP')}")
    else:
        print(f"EUS: {sample('EUS')}")
