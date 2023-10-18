from dgl.nn import GATConv, GraphConv
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0., bias=True,
                 act=nn.ReLU(), last_act=nn.Identity(),):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.layer2 = nn.Linear(hidden_dim, out_dim, bias=bias)
        self.nonlin =act
        self.last_nonlin = last_act
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, h):
        h = self.layer1(h)
        h = self.nonlin(h)
        h = self.dropout(h)
        h = self.layer2(h)
        return self.last_nonlin(h)


class GAT(nn.Module):
    """
    Graph Attention Network Class
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, gat_params,
                 act=nn.ELU(), last_act=nn.Identity()):
        super(GAT, self).__init__()
        self.layer1 = GATConv(in_dim, hidden_dim, num_heads, **gat_params)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = GATConv(hidden_dim * num_heads, out_dim, 1, **gat_params)
        self.act = act
        self.last_act = last_act

    def forward(self, graph, h):
        h = self.layer1(graph, h)
        # concatenate
        h = h.flatten(1)
        h = self.act(h)
        h = self.layer2(graph, h)
        return self.last_act(h.squeeze())
    

class GCNN_2L(nn.Module):
    """
    2-layer Graph Convolutional Neural Network Class as in Kipf
    """
    def __init__(self, in_dim, hidden_dim, out_dim, act=nn.ELU(), last_act=nn.Identity(),
                 norm='both', bias=True, dropout=0):
        super(GCNN_2L, self).__init__()
        self.layer1 = GraphConv(in_dim, hidden_dim, bias=bias, norm=norm)
        self.layer2 = GraphConv(hidden_dim, out_dim, bias=bias, norm=norm)
        self.dropout = nn.Dropout(p=dropout)
        self.act = act
        self.last_act = last_act

    def forward(self, graph, h):
        h = self.layer1(graph, h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.layer2(graph, h)
        return self.last_act(h)


# class FixedGFGNN(nn.Module):
#     """
#     Class for a Greapg Neural Network that replaces the classical normalized A by some given graph
#     fitler
#     """
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
