# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 11:39:59 2020

@author: josea
"""
import string
import math
from collections import Counter
from itertools import product
import pandas as pd
import torch


class CharNGram(object):
    """A character n-gram model trained on a list of words.

    Concepts from Jurafsky, D., & Martin, J.H. (2019). Speech and Language
    Processing. Stanford Press. https://web.stanford.edu/~jurafsky/slp3/

    This class is not optimized for large ngram models, use with caution
    for models of order 5 and above.
    """

    def __init__(self, data, n, laplace=1, SOS_token='<s>', EOS_token='</s>'):
        """Data should be iterable of words

        Args:
            data (List[str]): dataset from which to create the ngram model
            n (int): order of the model. Should be larger than 0
            laplace (int): additive smoothing factor for unseen combinations
                Default 1
            SOS_token (str): Start-of-Sequence token
            EOS_token (str): End-of-Sequence token
        """
        assert (n > 0), n
        self.n = n
        self.laplace = laplace
        self.SOS_token = SOS_token
        self.EOS_token = EOS_token
        self.data = data
        self.vocab = list(string.ascii_lowercase) + [SOS_token, EOS_token]
        self.processed_data = self._preprocess(self.data, n)
        self.ngrams = self._split_and_count(self.processed_data, self.n)
        self.model = self._smooth()

    def _preprocess(self, data, n):
        """Private method to preprocess a dataset of documents

        Args:
            data (List[str]): documents to be processed
            n (int): order of ngram model for processing
        Returns:
            new_data (List[str]): preprocessed data
        """
        new_data = []
        for word in data:
            new_data.append(self.process_word(word, n))
        return new_data

    def process_word(self, word, n):
        """Adds SOS and EOS tokens with padding

        Adds padding of SOS_tokens and EOS_tokens to each document
            padding size = n-1 for n > 1

        Args:
            word (str): word to be padded
            n (int): order of ngram model
        Returns:
            padded word (List[str])
        """
        pad = max(1, n-1)
        return [self.SOS_token] * pad +\
            list(word.lower()) +\
            [self.EOS_token]

    def _split_word(self, word, n):
        """Private generator to handle moving window over word of size n"""
        for i in range(len(word) - n + 1):
            yield tuple(word[i:i+n])

    def _split_and_count(self, data, n):
        """Private method to create ngram counts

        Args:
            data (List[str]): preprocessed data
            n (int): order of ngram model
        Returns:
            cntr (Counter): count of each ngram in data
        """
        cntr = self._initialize_counts(n)
        for word in data:
            for ngram in self._split_word(word, n):
                cntr[ngram] += 1
        return cntr

    def _initialize_counts(self, n):
        """Private method to initialize the ngram counter

        Accounts for unseen tokens by taking the product of the vocabulary

        Args:
            n (int): order of ngram model
        Returns:
            cntr (Counter): initialized counter of 0s for each plausible ngram
        """
        def is_plausible(permutation):
            if self.SOS_token not in permutation and \
                    self.EOS_token not in permutation:
                return True
            if self.SOS_token in permutation and\
                    self.EOS_token in permutation:
                return False
            n = len(permutation)

            if self.EOS_token in permutation[:-1]:
                return False
            if self.SOS_token in permutation[1:]:
                return False
            flg = False
            cnt = 0
            for i in range(n-1, -1, -1):
                if self.SOS_token == permutation[i]:
                    flg = True
                    cnt += 1
                else:
                    if flg:
                        return False
            if cnt == n:
                return False

            flg = False
            cnt = 0
            for i in range(n):
                if self.EOS_token == permutation[i]:
                    flg = True
                    cnt += 1
                else:
                    if flg:
                        return False
            return True
            if cnt == n:
                return False

        cntr = Counter()
        for perm in product(self.vocab, repeat=n):
            if is_plausible(perm):
                cntr[tuple(perm)] = 0
        return cntr

    def _smooth(self):
        """Private method to convert counts to probabilities using
        additive Laplace smoothing

        Returns:
            cntr (Counter): normalized probabilities of each ngram in data
        """
        if self.n == 1:
            s = sum(self.ngrams.values())
            return Counter({key: val/s for key, val in self.ngrams.items()})
        else:
            vocab_size = len(self.vocab)-1

            ret = self.ngrams.copy()

            m = self.n - 1
            m_grams = self._split_and_count(self.processed_data, m)

            for ngram, value in self.ngrams.items():
                m_gram = ngram[:-1]
                m_count = m_grams[m_gram]
                ret[ngram] = (value + self.laplace) /\
                    (m_count + self.laplace * vocab_size)

            return ret

    def to_txt(self, filepath):
        """Saves model to disk as a tab separated txt file"""
        with open(filepath, 'w') as file:
            for ngram, value in self.model.items():
                file.write(f"{' '.join(ngram)}\t{value}\n")

    def from_txt(self, filepath):
        """Reads model from a tab separated txt file"""
        with open(filepath, 'r') as file:
            data = file.readlines()
        model = Counter()
        for ngram, value in data.split('\t'):
            model[tuple(ngram.split(' '))] = value
        return model

    def to_df(self):
        """Creates a DataFrame from Counter of ngrams

        Warning: Do not use with ngrams of order >= 5

        Returns:
            df (pandas.DataFrame): dataframe of normalized probabilities
                shape [n_plausible_ngrams, len(vocab)]
        """
        idxs, cols = set(), set()
        for k in self.model.keys():
            idxs.add(' '.join(k[:-1]))
            cols.add(k[-1])
        df = pd.DataFrame(data=0.0,
                          index=sorted(list(idxs)),
                          columns=sorted(list(cols)))
        for ngram, value in self.model.items():
            cntx = ' '.join(ngram[:-1])
            trgt = ngram[-1]
            df.loc[cntx, trgt] = value
        return df.fillna(0.0)

    def get_single_probability(self, word, log=False):
        """Calculates the probability (likelihood) of a word given the ngram
        model

        Args:
            word (str or List[str]): target word
            log (bool): whether to get loglikelihood instead of probability
        Returns:
            prob (float): probability of the word given the ngram model
        """
        if isinstance(word, str):
            word = self.process_word(word, self.n)
        n = len(word)
        prob = 0.0 if log else 1.0
        for ngram in self._split_word(word, self.n):
            if ngram not in self.model.keys():
                n -= 1
                print(ngram)
                continue
            p = self.model[ngram]
            if log:
                prob += math.log(p)
            else:
                prob *= p
        return prob / n

    def perplexity(self, data):
        """Calculates the perplexity of an entire dataset given the model

        Perplexity is the inverse probability of the dataset, normalized
        by the number of words

        To avoid numeric overflow due to multiplication of probabilities,
        the probabilties are log-transformed and the final score is then
        exponentiated. Thus:

            Perplexity = exp(-(1/N) * sum(probs))

        where N is the number of words and probs is the vector of probabilities
        for each word in the dataset.

        Lower perplexity is equivalent to higher probability of the data given
        the ngram model.

        Args:
            data (\List[str]): datset of words
        Returns:
            perplexity (float): perplexity of the dataset given the ngram model
        """
        test_tokens = self._preprocess(data, self.n)
        N = len(test_tokens)

        probs = 0.0
        for word in test_tokens:
            probs += self.get_single_probability(word, log=True)

        return math.exp((-1/N) * probs)

    def get_distribution_from_context(self, context):
        """Get the multinomial distribution for the next character given a
        context

        Args:
            context (str or List[str]): context of variable length
        Returns:
            dist (dict): probability distribution of the next letter
        """
        m = len(context)
        if m < self.n-1:
            context = [self.SOS_token] * (self.n-m-1) + list(context)
        elif m > self.n-1:
            context = list(context[-self.n+1:])
        context = list(context)
        dist = {v: 0 for v in self.vocab}
        for v in self.vocab:
            dist[v] = self.model[tuple(context + [v])]
        del dist[self.SOS_token]
        return dist

    def _next_candidate(self, prev, without=[]):
        """Private method to select next candidate from previous context

        Candidates are selected at random from a multinomial distribution
        weighted by the probability of next token given context.

        Args:
            prev (Tuple[str]): previous context
        Returns:
            letter (str): selected next candidate
            prob (float): probability of next candidate given context
        """
        letters = self.get_distribution_from_context(prev)
        letters = {l: prob for l, prob in letters.items() if l not in without}
        letters, probs = list(letters.keys()), list(letters.values())
        topi = torch.multinomial(torch.FloatTensor(probs), 1)[0].item()
        return letters[topi], probs[topi]

    def generate_words(self, num, min_len=3, max_len=10, without=[]):
        """Generates a number of words by sampling from the ngram model

        Generator method.

        Args:
            num (int): number of words to generate
            min_len (int): minimum length of the words
            max_len (int): maximum length of the words
            without (List[str]): list of tokens to ignore during selection
        Yields:
            word (str): generated word
        """
        for i in range(num):
            word, prob = [self.SOS_token] * max(1, self.n-1), 1
            while word[-1] != self.EOS_token:
                prev = () if self.n == 1 else tuple(word[-self.n+1:])
                blacklist = [self.EOS_token] if len(word) < min_len else []
                next_token, next_prob = self._best_candidate(
                    prev, i, without=blacklist)
                word.append(next_token)
                prob *= next_prob
                if len(word) >= max_len:
                    word.append(self.EOS_token)
            word = [w for w in word if w not in [
                self.SOS_token, self.EOS_token]]
            yield ''.join(word), -1/math.log(prob)

    def __len__(self):
        return len(self.ngrams)

    def __str__(self):
        return f"<{self.n}-gram model(size={len(self)})>"
