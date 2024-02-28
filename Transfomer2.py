import torch
import torch.nn.functional as F
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
from GRACE import Encoder, GRACE, drop_feature
from torch_geometric.nn import GCNConv
#from torch_geometric.utils import dropout_adj
from torch_geometric.utils import dropout_edge
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
        elif modelName=='GRACE':
            self.drop_edge_rate_1 = 0.2
            self.drop_edge_rate_2 = 0.0
            self.drop_feature_rate_1 = 0.3
            self.drop_feature_rate_2 = 0.1
            self.activation='prelu'
            self.base_model='GraphConvolution'
            self.new_model='GCNN'
            activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[self.activation]
            base_model = ({'GCNConv': GCNConv,'GraphConvolution':GraphConvolution})[self.base_model]
            new_model = ({'GCNConv': GCNConv, 'GraphConvolution': GraphConvolution,'GCNN':GCNN})[self.new_model]
            encoder = Encoder(args['embedding_size'], args.hidden_size, activation,base_model=base_model,new_model=new_model, k=args.nums_layers)
            # self.Tconv_forward=NET[self.NETname]( encoder, int(args['embedding_size']/(2**(args.nums_layers))), int(args['embedding_size']/(2**(args.nums_layers))), tau=args.tau)
            self.Tconv_forward = NET[self.NETname]( encoder, int( hidden ), int( hidden ), tau=args.tau )
        self.sublayer4 = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNorm(hidden)

    def forward(self, x, mask, inputP):
        # x = self.sublayer4(x, lambda _x: self.Tconv_forward.forward(_x, None, inputP))
        # x = self.norm(x)
        # return self.dropout(x)
        if self.NETname not in ['GRACE']:
            x = self.sublayer4(x, lambda _x: self.Tconv_forward.forward(_x, None, inputP))
            x = self.norm(x)
            return self.dropout( x ), 0
        else:
            shape = inputP.shape
            if self.drop_edge_rate_1 > 0:
                A1 = []
                for i in range( 0, shape[0] ):
                    #edge_index_1 = dropout_adj( inputP[i]._indices(), p=self.drop_edge_rate_1)[0]
                    edge_index_1 = dropout_edge( inputP[i]._indices(), p=self.drop_edge_rate_1)[0]
                    v = torch.ones(edge_index_1.shape[1]).cuda()
                    A1.append(torch.sparse_coo_tensor( edge_index_1, v, (shape[1], shape[2])))
                A1=torch.stack(A1,dim=0)
                # edge_index_1=A._indices()
                # print("A",A,edge_index_1.shape)
            else:
                # edge_index_1 = inputP._indices()
                A1 = inputP
            if self.drop_edge_rate_2>0:
                A2 = []
                for i in range( 0, shape[0] ):
                    #edge_index_2 = dropout_adj( inputP[i]._indices(), p=self.drop_edge_rate_2 )[0]
                    edge_index_2 = dropout_edge( inputP[i]._indices(), p=self.drop_edge_rate_2 )[0]
                    v = torch.ones( edge_index_2.shape[1] ).cuda()
                    A2.append( torch.sparse_coo_tensor( edge_index_2, v, (shape[1], shape[2]) ) )
                A2 = torch.stack( A2, dim=0 )
                # edge_index_2 = A._indices()
                # print("A", A, edge_index_2.shape)
            else:
                # edge_index_2 = inputP._indices()
                A2=inputP

            # edge_index_1 = dropout_adj( inputP[0]._indices(), p=self.drop_edge_rate_1)[0]
            # edge_index_2 = dropout_adj( inputP[0]._indices(), p=self.drop_edge_rate_2)[0]
            x_1 = drop_feature( x, self.drop_feature_rate_1 )
            x_2 = drop_feature( x, self.drop_feature_rate_2)
            z1 = self.Tconv_forward.forward( x_1, A1 )
            z2 = self.Tconv_forward.forward( x_2, A2 )
            loss=self.Tconv_forward.loss( z1, z2, batch_size=shape[0])
            print("loss",loss)
            return z2,loss
