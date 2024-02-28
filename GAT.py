import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat,nheads, dropout=0.1, alpha=0.2):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        nhid=int(nfeat/nheads)
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nhid * nheads, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x,left, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        #x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x


class SpGAT(nn.Module):
    def __init__(self, nhid,nheads,dropout=0.1, alpha=0.2):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout
        nfeat = nhid / nheads
        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # self.out_att = SpGraphAttentionLayer(nhid * nheads,
        #                                      nhid * nheads,
        #                                      dropout=dropout,
        #                                      alpha=alpha,
        #                                      concat=False)

    def forward(self, x, left, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        #x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.out_att(x, adj))
        # x=F.log_softmax(x, dim=1)
        return x

