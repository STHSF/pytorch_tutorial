#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: 1.0
@author: li
@file: bilstm_attention.py
@time: 2020/1/15 11:30 上午
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, encoder_outputs, x_mask):
        energy = self.projection(encoder_outputs).squeeze(-1)
        energy.masked_fill_(x_mask, -float('inf'))
        weights = F.softmax(energy, dim=1)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs


class C2ResRNN(nn.Module):
    def __init__(self, vocab_size, config):
        super(C2ResRNN, self).__init__()
        self.dropout_rate = config.dropout_rate
        self.relation_size = config.relation_size
        self.config = config
        self.dropout = nn.Dropout(config.dropout_rate)
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=config.embedding_dim)
        self.cnn_dim = 100
        self.window_sizes = [3, 5, 7]
        self.embedding_dim = config.embedding_dim
        self.convs = nn.ModuleList([nn.Sequential(nn.Conv1d(in_channels=self.embedding_dim,
                                                            out_channels=self.cnn_dim,
                                                            kernel_size=h,
                                                            padding=h // 2),
                                                  nn.ReLU())
                                    for h in self.window_sizes])
        self.bilstm1 = nn.LSTM(input_size=config.embedding_dim + len(self.window_sizes) * config.cnn_dim,
                               hidden_size=config.hidden_size,
                               num_layers=1,
                               bias=True,
                               batch_first=True,
                               bidirectional=True)
        self.bilstm2 = nn.LSTM(input_size=config.hidden_size * 2,
                               hidden_size=config.hidden_size,
                               num_layers=1,
                               bias=True,
                               batch_first=True,
                               bidirectional=True)
        self.attention = Attention(config.hidden_size * 4)
        self.fc1 = nn.Linear(in_features=config.hidden_size * 4,
                             out_features=config.feature_size)
        self.fc2 = nn.Linear(in_features=config.feature_size,
                             out_features=config.relation_size)

    def forward(self, x, pos1=None, pos2=None, mask=None, length=None):
        x_mask = x.eq(self.config.PAD_ID)
        embed_x = self.embedding(x)
        embed_x = self.dropout(embed_x)
        embed_x = embed_x.transpose(1, 2)
        cnn_out_list = [conv(embed_x) for conv in self.convs]
        cnn_output = torch.cat(cnn_out_list, dim=1)
        embed_x = torch.cat([embed_x, cnn_output], dim=1)
        embed_x = embed_x.transpose(1, 2)
        lstm_out1, _ = self.bilstm1(embed_x)
        lstm_out2, _ = self.bilstm2(lstm_out1)
        lstm_out = torch.cat([lstm_out1, lstm_out2], dim=2)
        attn_in = lstm_out
        attn_out = self.attention(attn_in, x_mask)
        out = attn_out.view(-1, attn_out.size(1))
        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class train_config(object):
    dropout_rate = 0.1
    relation_size = 10
    embedding_dim = 200
    cnn_dim = 200
    hidden_size = 10
    feature_size = 10


model = C2ResRNN(500, config=train_config)

print(model)
