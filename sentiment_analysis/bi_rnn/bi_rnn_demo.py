#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: li
@file: bi_rnn_demo.py
@time: 2019/12/9 9:16 下午
"""
import os
import gensim
from itertools import chain

word_to_idx = {}
def read_imdb(path, seg='train'):
    pos_or_neg = ['pos', 'neg']
    data = []
    for label in pos_or_neg:
        files = os.listdir(os.path.join(path, seg, label))
        for file in files:
            with open(os.path.join(path, seg, label, file), 'r', encoding='utf8') as rf:
                review = rf.read().replace('\n', '')
                if label == 'pos':
                    data.append([review, 1])
                elif label == 'neg':
                    data.append([review, 0])
    return data


def tokenizer(text):
    return [tok.lower() for tok in text.split(' ')]


def encode_samples(tokenized_samples, vocab):
    features = []
    for sample in tokenized_samples:
        feature = []
        for token in sample:
            if token in word_to_idx:
                feature.append(word_to_idx[token])
            else:
                feature.append(0)
        features.append(feature)
    return features


def pad_samples(features, maxlen=500, PAD=0):
    padded_features = []
    for feature in features:
        if len(feature) >= maxlen:
            padded_feature = feature[:maxlen]
        else:
            padded_feature = feature
            while(len(padded_feature) < maxlen):
                padded_feature.append(PAD)
        padded_features.append(padded_feature)
    return padded_features


wvmodelwvmodel = gensim.models.KeyedVectors.load_word2vec_format('test_word.txt',
                                                          binary=False, encoding='utf-8')


train_data = read_imdb('aclImdb')
test_data = read_imdb('aclImdb', 'test')

train_tokenized = []
test_tokenized = []
for review, score in train_data:
    train_tokenized.append(tokenizer(review))
for review, score in test_data:
    test_tokenized.append(tokenizer(review))

vocab = set(chain(*train_tokenized))
vocab_size = len(vocab)


word_to_idxword_to  = {word: i+1 for i, word in enumerate(vocab)}
word_to_idx['<unk>'] = 0
idx_to_word = {i+1: word for i, word in enumerate(vocab)}
idx_to_word[0] = '<unk>'