from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

class BaseLine(nn.Module):
    def __init__(self, hidden_dim, dropout_rate, vocab_size, embedding_dim, bi_rnn_layers, uni_rnn_layers, pre_trained_embedding=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.embedding_dim = embedding_dim

        if pre_trained_embedding is None:
            self.vocab_size = vocab_size
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=1)
        else:
            self.embedding = nn.Embedding.from_pretrained(pre_trained_embedding, freeze=False, padding_idx=1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.bi_rnn = nn.GRU(self.embedding_dim, int(self.hidden_dim / 2), num_layers=bi_rnn_layers, batch_first=False, bidirectional=True)
        self.uni_rnn = nn.GRU(self.hidden_dim, self.hidden_dim, num_layers=uni_rnn_layers, batch_first=False)
        self.linear = nn.Linear(self.hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (sentence_len, batch_size)
        x = self.embedding(x)
        # x: (sentence_len, batch_size, embedding_dim)
        x = self.dropout(x)
        x, _ = self.bi_rnn(x)
        # x: (sentence_len, batch_size, hidden_dim)
        x = self.relu(x)
        x = self.dropout(x)
        x, _ = self.uni_rnn(x)
        x = self.dropout(x)
        # x: (sentence_len, batch_size, hidden_dim)
        x, _ = torch.max(x, 0)
        # x: (batch_size, hidden_dim)
        x = self.linear(x)
        x = self.sigmoid(x).squeeze()
        return x
