import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret

# 模型定义
# 读者可以自己对GCN模型结构进行修改和实验
# class Encoder(torch.nn.Module):#in_channels--64 out_channels--32
#     def __init__(self, in_channels: int, out_channels: int, activation,
#                  base_model=GCNConv, k: int = 2):
#         super(Encoder, self).__init__()
#         self.base_model = base_model
#
#         assert k >= 2
#         self.k = k
#         self.conv = [base_model(in_channels, 2 * out_channels)]
#         for _ in range(1, k-1):
#             self.conv.append(base_model(2 * out_channels, 2 * out_channels))
#         self.conv.append(base_model(2 * out_channels, out_channels))
#         self.conv = nn.ModuleList(self.conv)
#
#         self.activation = activation
#
#     def forward(self, x, A):
#         for i in range(self.k):
#             x = self.activation(self.conv[i](A, x))
#         return x
class Encoder(torch.nn.Module):#in_channels--64 out_channels--32
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, new_model=GCNConv, k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model

        # assert k >= 2
        self.k = k
        # self.conv.append( new_model( in_channels, in_channels ) )
        self.conv=[]
        for _ in range(0, k):
            self.conv.append(new_model(in_channels,in_channels))
#            self.conv.append(base_model(in_channels,out_channels))
#            in_channels=out_channels
            # out_channels=int(out_channels/2)
        # self.conv.append(new_model(out_channels,out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    def forward(self, x, A):
#        for i in range(0,2*self.k,2):
        for i in range(0,self.k):
#            print("i:",i)
            x = self.activation(self.conv[i]( x,None, A))
#            x = self.activation( self.conv[i+1]( A, x ) )
        return x

class GRACE(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5):
        super(GRACE, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau
        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())#calculating Cosing by normalize and mm

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))#intra-view negative pairs
        between_sim = f(self.sim(z1, z2))#inter-view negative pairs

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          mean: bool = True, batch_size: int = 0):
        loss = torch.zeros(1, requires_grad=True).cuda()
        losses=[]
        for i in range(batch_size):
            l1=self.semi_loss( z1[i], z2[i])
            l2=self.semi_loss( z2[i], z1[i])
            ret = (l1 + l2) * 0.5
            ret = ret.mean().item() if mean else ret.sum()
#            loss=loss.add(ret)
            losses.append(ret)
        losses=torch.Tensor(losses)
#        print("losses",losses)
        losses=losses.mean() if mean else losses.sum()
        loss=loss.add(losses)
#        print("loss",loss)     
        return  loss
        # return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
            ret = (l1 + l2) * 0.5
            ret = ret.mean() if mean else ret.sum()
        else:
            ret = self.batched_semi_loss(h1, h2, True, batch_size)
        return ret


def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(2), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, :, drop_mask] = 0
    return x

def drop_edge(A, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
