import torch.nn as nn
from gcnn import GCNN
from GGNN import GGNN
from GAT import GAT
from GGAT import GGAT
from GCN import GcnNet
from Multihead_Attention import MultiHeadedAttention
from SubLayerConnection import SublayerConnection
from DenseLayer import DenseLayer
from LayerNorm import LayerNorm
class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout,modelName):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        #elif modelName=='GcnNetAT':
        #     self.Tconv_forward = GcnNetAT(hidden, attn_heads)
        # elif modelName=='GcnNetSA':
        #     self.Tconv_forward = GcnNetSA(hidden, attn_heads)
        # elif modelName=='GcnNetSA4':
        #     self.Tconv_forward = GcnNetSA4(hidden, attn_heads)
        if modelName=='GGNN':
            self.Tconv_forward = GGNN(hidden)
        # elif modelName=='GGNNSA':
        #     self.Tconv_forward = GGNNSA(hidden)
        # elif modelName=='GGNNSA4':
        #     self.Tconv_forward = GGNNSA4(hidden)
        elif modelName=='GGAT':
            self.Tconv_forward = GGAT(hidden,attn_heads)
        elif modelName=='GcnNet':
            self.Tconv_forward = GcnNet(hidden)
        # elif modelName=='NewGAT':
        #     self.Tconv_forward = NewGAT( hidden, attn_heads, dropout=0.1, alpha=0.2 )
        elif modelName=='GAT':
            self.Tconv_forward = GAT(hidden, attn_heads,dropout=0.1, alpha=0.2)
        # elif modelName=='SpGAT':
        #     self.Tconv_forward = SpGAT( nhid=hidden, nheads=attn_heads, dropout=0.1, alpha=0.2 )
        elif modelName=='GCNN':
            self.Tconv_forward = GCNN(dmodel=hidden)
        self.sublayer4 = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNorm(hidden)

    def forward(self, x, mask, inputP):
#        print("x.device",x.device)
        #x = self.sublayer1(x, lambda _x: self.attention1.forward(_x, _x, _x, mask=mask))
        #x = self.sublayer2(x, lambda _x: self.combination.forward(_x, _x, pos))
        #x = self.sublayer3(x, lambda _x: self.combination2.forward(_x, _x, charem))
        x = self.sublayer4(x, lambda _x: self.Tconv_forward.forward(_x, None, inputP))
        x = self.norm(x)
        return self.dropout(x)
