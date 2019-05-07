import torch
from torch import nn


class MeanAggregator(nn.Module):
    def __init__(self, _):
        super(MeanAggregator, self).__init__()

    def forward(self, feats):
        return feats.average(dim=1)


class LSTMAggregator(nn.Module):
    def __init__(self, feat_dim):
        super(LSTMAggregator, self).__init__()
        out_size = hidden_size = feat_dim
        self.rnn = nn.LSTM(feat_dim, hidden_size, out_size)

    def forward(self, feats, shuffle=False):
        # TODO: shuffle option
        rst, _ = self.rnn(feats)
        return rst


class PoolingAggregator(nn.Module):
    def __init__(self):
        super(PoolingAggregator, self).__init__()

    def forward(self, feats):
        raise NotImplementedError
