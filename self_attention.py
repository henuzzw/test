import torch
from torch import nn

class selfattention( nn.Module ):
    def __init__(self, kq_size,v_size,flag=False):
        super(selfattention,self).__init__()
        self.flag=flag
        self.kq_size =kq_size
        self.v_size  =v_size
        self.dropout = 0.1
        self.alpha = 0.2
        self.query = nn.Linear( self.kq_size, self.v_size)
        self.key   = nn.Linear( self.kq_size, self.v_size)
        self.value = nn.Linear( self.v_size, self.v_size)
        self.gamma = nn.Parameter(torch.zeros( 1 ))  # gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax( dim=-1 )

    def forward(self, input):
        batch_size, node_size, embedding_size  = input.shape
        if self.flag==False:
            kqinput=torch.tensor(input[:,:,-1*self.kq_size:]).float().reshape(batch_size,node_size,self.kq_size)
        else:
            kqinput=torch.tensor(input[:,:,:self.kq_size]).float().reshape(batch_size,node_size,self.kq_size)
        # input: B, n, 4 -> q: B, w,32,
        q = self.query( kqinput ).view( batch_size,-1, self.v_size)
        # input: B, n, 4 -> q: B, 32,w,
        k = self.key( kqinput ).view( batch_size,-1, self.v_size).permute( 0, 2, 1 )
        # input: B, n, e -> q: B, n,e,
        v = self.value( input ).view( batch_size,node_size, self.v_size)
        attn_matrix = torch.bmm( k,q)
        attn_matrix = self.softmax( attn_matrix )
        out = torch.bmm(v,attn_matrix)
        return self.gamma * out + input

