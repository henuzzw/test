import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from layers import GraphAttentionLayer
from self_attention import selfattention
class GraphConvolution( nn.Module ):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """图卷积：L*X*\theta
        Args:
        ----------
            input_dim: int
                节点输入特征的维度
            output_dim: int
                输出特征维度
            use_bias : bool, optional
                是否使用偏置
        """
        super( GraphConvolution, self ).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter( torch.Tensor( input_dim, output_dim ) )
        if self.use_bias:
            self.bias = nn.Parameter( torch.Tensor( output_dim ) )
        else:
            self.register_parameter( 'bias', None )
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_( self.weight )
        if self.use_bias:
            init.zeros_( self.bias )

    def forward(self, adjacency,input_feature):
        """邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法
        Args:
        -------
            adjacency: torch.sparse.FloatTensor
                邻接矩阵
            input_feature: torch.Tensor
                输入特征
        """
#        adjacency = adjacency.to_dense()
        support = torch.matmul( input_feature, self.weight )
#        output = torch.matmul( adjacency, support )
        x=adjacency.shape
        y=support.shape
        #print("CUDA",adjacency[0].shape)#adjacency[0].device,support[0].device)
        output=torch.matmul(adjacency[0], support[0]).reshape(-1,x[1],y[2])
        for i in range(1,x[0]):
            output=torch.cat([output,torch.matmul(adjacency[i], support[i]).reshape(-1,x[1],y[2])],dim=0)
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str( self.input_dim ) + ' -> ' + str( self.output_dim ) + ')'


# 模型定义
# 读者可以自己对GCN模型结构进行修改和实验
class GcnNet( nn.Module ):
    """
    定义一个包含两层GraphConvolution的模型
    """

    def __init__(self, input_dim=32):
        super( GcnNet, self ).__init__()
        self.gcn1 = GraphConvolution( input_dim, 16 )
        self.gcn2 = GraphConvolution( 16, input_dim )

    def forward(self, feature,left,adjacency ):
        h = F.relu( self.gcn1( adjacency, feature ) )
        logits = self.gcn2( adjacency, h )
        return logits

class GcnNetAT( nn.Module ):
    """
    定义一个包含两层GraphConvolution的模型
    """

    def __init__(self, input_dim=32,nheads=1):
        super( GcnNetAT, self ).__init__()
        self.gcn1 = GraphConvolution( input_dim, 16 )
        self.gcn2 = GraphConvolution( 16, input_dim )
        nhid=int(input_dim/nheads)
        self.attentions = [GraphAttentionLayer(input_dim, nhid, dropout=0.1, alpha=0.2, concat=True) for
                           _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
    def forward(self, feature,left,adjacency ):
        feature=torch.cat([att(feature, adjacency) for att in self.attentions], dim=-1)
        h = F.relu( self.gcn1( adjacency, feature ) )
        logits = self.gcn2( adjacency, h )
        return logits
class GcnNetSA4( nn.Module ):
    """
    定义一个包含两层GraphConvolution的模型
    """

    def __init__(self, input_dim=32,nheads=1):
        super( GcnNetSA4, self ).__init__()
        self.gcn1 = GraphConvolution( input_dim, 16 )
        self.gcn2 = GraphConvolution( 16, input_dim )
        self.self_att1 = selfattention(4, input_dim)
        #self.self_att0 = selfattention(28, input_dim, flag=True)
    def forward(self, feature,left,adjacency ):
        state0 = self.self_att1(feature)
        #state1 = self.self_att1(feature)
        #feature = state0.add_(state1)
        #h = F.relu( self.gcn1( adjacency, feature ) )
        h = F.relu( self.gcn1( adjacency, state0 ) )
        logits = self.gcn2( adjacency, h )
        return logits
class GcnNetSA( nn.Module ):
    """
    定义一个包含两层GraphConvolution的模型
    """

    def __init__(self, input_dim=32,nheads=1):
        super( GcnNetSA, self ).__init__()
        self.gcn1 = GraphConvolution( input_dim, 16 )
        self.gcn2 = GraphConvolution( 16, input_dim )
        self.self_att1 = selfattention(4, input_dim)
        self.self_att0 = selfattention(28, input_dim, flag=True)
    def forward(self, feature,left,adjacency ):
        state0 = self.self_att1(feature)
        state1 = self.self_att1(feature)
        feature = state0.add_(state1)
        h = F.relu( self.gcn1( adjacency, feature ) )
        logits = self.gcn2( adjacency, h )
        return logits