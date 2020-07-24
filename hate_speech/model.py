from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

class BaseLine(nn.Module):
    def __init__(self, hidden_dim, filter_size, dropout_rate, vocab_size, embedding_dim, num_layers, padding, pre_trained_embedding=None):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.filter_size = filter_size
        self.dropout_rate = dropout_rate
        self.embedding_dim = embedding_dim

        if pre_trained_embedding is None:
            self.vocab_size = vocab_size
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=1)
        else:
            self.embedding = nn.Embedding.from_pretrained(pre_trained_embedding, freeze=False, padding_idx=1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_rate)
        # self.conv1d = nn.Conv1d(self.embedding_dim, self.hidden_dim, self.filter_size, padding=padding)
        self.bi_rnn = nn.GRU(self.embedding_dim, int(self.hidden_dim / 2), num_layers=num_layers, batch_first=False, bidirectional=True)
        self.uni_rnn = nn.GRU(self.hidden_dim, self.hidden_dim, num_layers=num_layers, batch_first=False)
        # self.max_pool = nn.AdaptiveAvgPool2d((1, self.hidden_dim))
        self.linear = nn.Linear(self.hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (sentence_len, batch_size)
        x = self.embedding(x).transpose(0, 1).transpose(1, 2)
        # x: (batch_size, embedding_dim, sentence_len)
        # x = self.conv1d(x).transpose(1, 2).transpose(0, 1)
        x = x.transpose(1, 2).transpose(0, 1)
        # x: (sentence_len, batch_size, embedding_dim)
        x = self.relu(x)
        x = self.dropout(x)
        # x_res = x
        # x_res: (sentence_len, batch_size, embedding_dim)
        x, _ = self.bi_rnn(x)
        # x: (sentence_len, batch_size, hidden_dim)
        # x, _ = self.uni_rnn(x + x_res)
        x, _ = self.uni_rnn(x)
        # x: (sentence_len, batch_size, hidden_dim)
        x = self.dropout(x)
        x, _ = torch.max(x, 0)
        # x: (batch_size, hidden_dim)
        x = self.linear(x)
        x = self.sigmoid(x).squeeze()
        return x
