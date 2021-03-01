# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 14:18:44 2020

@author: josea
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLOUD(nn.Module):
    """
    The CLOUD architecture learns the orthotactic patterns of a language
    by sequentially predicting the next character.

    Characters are identified by their index (e.g., a = 0, b = 1, ...).
    The character one-hot vectors go through the recurrent layer, producing hidden representations
    The model uses the hidden representations to output the probability distribution of the next character.
    """

    def __init__(self, char_vocab_size, n_embedd=32, n_hidden=128, n_layers=1, drop_p=0.0, hidden_type='LSTM', pad_idx=None):
        """
        Args:
            char_vocab_size (int): The number of characters in the vocabulary (alphabet + special characters)
            n_hidden (int): The size of the RNN's hidden representations (Default 128)
            n_layers (int): Number of RNN layers (Default 1)
            drop_p (float): Dropout between RNN layers if n_layers > 1 (Default 0.2)
            hidden_type (str): Whether t
            o use LSTM or GRU cells (Default 'LSTM')
            pad_idx (int): Index to ignore in Embedding layer
        """
        super(CLOUD, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.hidden_type = hidden_type
        self.char_vocab_size = char_vocab_size
        self.pad_idx = pad_idx

        self.E = nn.Embedding(num_embeddings=char_vocab_size,
                              embedding_dim=n_embedd,
                              padding_idx=pad_idx)

        if hidden_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=n_embedd,
                hidden_size=n_hidden,
                num_layers=n_layers,
                dropout=drop_p if n_layers > 1 else 0,
                batch_first=True
            )
        elif hidden_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=n_embedd,
                hidden_size=n_hidden,
                num_layers=n_layers,
                dropout=drop_p if n_layers > 1 else 0,
                batch_first=True
            )
        else:
            raise ValueError('Only "GRU" or "LSTM" accepted as hidden_type.')

        self.fc = nn.Linear(n_hidden, char_vocab_size)

    def forward(self, X, X_lengths, hidden, drop_rate=0.0, max_length=None):
        """
        Args:
            X (torch.Tensor): Tensor representations of character sequences (words)
            X_lengths (torch.Tensor): Tensor with the lengths of each sequence
            hidden (torch.Tensor): Hidden representations at t0 (init_hidden)
            drop_rate (float): Dropout between RNN layer and output layer (Defaults to 0.0 during evaluation)
            max_length (int): Maximum length of padded sequences (avoids errors). 
                If None the pad_packed_sequence function will infer the max length
        """
        X = self.E(X)

        X = nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True)

        # Hidden layer
        out, hidden = self.rnn(X, hidden)

        out_rnn, _ = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, padding_value=self.pad_idx, total_length=max_length)

        # Reshape output
        batch_size, seq_size, n_hidden = out_rnn.shape
        out = out_rnn.contiguous().view(batch_size*seq_size, n_hidden)

        # Output layer
        out = self.fc(F.dropout(out, p=drop_rate))
        out.view(batch_size, seq_size, self.char_vocab_size)

        return out, out_rnn, hidden

    def init_hidden(self, batch_size, device='cpu'):
        """
        Initializes hidden representations for LSTM or GRU cells
        Args:
            batch_size (int): How many words to process at once
            device (torch.device): Where to store these tensors
        """
        hidden = torch.zeros(self.n_layers, batch_size,
                             self.n_hidden)
        hidden = hidden.to(device)

        if self.hidden_type == 'LSTM':
            cell = torch.zeros(self.n_layers, batch_size,
                               self.n_hidden)
            cell = cell.to(device)
            hidden = (hidden, cell)

        return hidden
