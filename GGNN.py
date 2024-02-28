import torch
import torch.nn as nn
from self_attention import selfattention
class Propogator(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """
    def __init__(self, state_dim):
        super(Propogator, self).__init__()


        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*2, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*2, state_dim),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim*2, state_dim),
            nn.Tanh()
        )

    def forward(self,state, state_cur, A):
        a_t= torch.bmm(A, state)
        # print(a_in.size(),a_out.size(),state_cur.size())
        a = torch.cat((a_t, state_cur), 2)

        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_t, r * state_cur), 2)
        h_hat = self.tansform(joined_input)

        output = (1 - z) * state_cur + z * h_hat

        return output


class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, state_dim):
        super(GGNN, self).__init__()
        self.state_dim = state_dim

        self.linear= nn.Linear(self.state_dim, self.state_dim)

        # Propogation Model
        self.propogator = Propogator(self.state_dim)

        # Output Model
        self.out = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.state_dim, 1)
        )
        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, prop_state,left, A):
        prop_state=self.linear(prop_state)
        prop_state = self.propogator(prop_state,prop_state, A )
        return prop_state

class GGNNSA(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, state_dim):
        super(GGNNSA, self).__init__()
        self.state_dim = state_dim

        self.linear= nn.Linear(self.state_dim, self.state_dim)

        # Propogation Model
        self.propogator = Propogator(self.state_dim)

        # Output Model
        self.out = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.state_dim, 1)
        )
        self.self_att1 = selfattention( 4, self.state_dim )
        self.self_att0 = selfattention( 28, self.state_dim, flag=True )
        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, prop_state,left, A):
        prop_state=self.linear(prop_state)
        state0 = self.self_att0(prop_state )
        state1 = self.self_att1(prop_state )
        state = state0.add_( state1)
        prop_state = self.propogator(state,prop_state, A)
        return prop_state

class GGNNSA4(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, state_dim):
        super(GGNNSA4, self).__init__()
        self.state_dim = state_dim

        self.linear= nn.Linear(self.state_dim, self.state_dim)

        # Propogation Model
        self.propogator = Propogator(self.state_dim)

        # Output Model
        self.out = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.state_dim, 1)
        )
        self.self_att1 = selfattention( 4, self.state_dim )
        self.self_att0 = selfattention( 28, self.state_dim, flag=True )
        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, prop_state,left, A):
        prop_state=self.linear(prop_state)
        state = self.self_att0(prop_state )
        prop_state = self.propogator(state,prop_state, A)
        return prop_state
