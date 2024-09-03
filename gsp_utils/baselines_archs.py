from dgl.nn import GATConv, GraphConv, SAGEConv, GINConv
import torch.nn as nn


class BaselineArch(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers=2, act=nn.ReLU(), l_act=nn.Identity(),
                 bias=True, dropout=0):
        super(BaselineArch, self).__init__()
        self.in_d = in_dim
        self.hid_d = hid_dim
        self.out_d = out_dim
        self.dropout = nn.Dropout(p=dropout)
        self.act = act
        self.l_act = l_act

        self.convs = self._create_conv_layers(n_layers, bias)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, A, X):
        for _, conv in enumerate(self.convs[:-1]):
            X = self.act(conv(A, X))
            X = self.dropout(X)

        X_out = self.convs[-1](A, X)
        return self.l_act(X_out)


# class MyGCNNLayer(nn.Module):
#     def __init__(self, in_dim, out_dim, bias):
#         super(MyGCNNLayer, self).__init__()
#         self.in_d = in_dim
#         self.out_d = out_dim
#         self._init_parameters(bias)

#     def _init_parameters(self, bias):
#         self.W = nn.Parameter(torch.empty((self.in_d, self.out_d)))
#         nn.init.xavier_uniform_(self.W)

#         if bias:
#             self.b = nn.Parameter(torch.empty(self.out_d))
#             nn.init.constant_(self.b.data, 0.)
#         else:
#             self.b = None

#     def forward(self, A, X):
#         X_out = A @ (X @ self.W)

#         if self.b is not None:
#             return X_out + self.b[None,:]
#         else:
#             return X_out


class GCNN(BaselineArch):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers=2, act=nn.ReLU(), l_act=nn.Identity(),
                 bias=True, dropout=0, norm='both'):
        self.norm = norm
        super(GCNN, self).__init__(in_dim, hid_dim, out_dim, n_layers=2, act=nn.ReLU(),
                                   l_act=nn.Identity(), bias=True, dropout=0)

    def _create_conv_layers(self, n_layers: int, bias: bool) -> nn.ModuleList:
        convs = nn.ModuleList()
        
        if n_layers > 1:
            convs.append( GraphConv(self.in_d, self.hid_d, bias=bias, norm=self.norm) )
            
            for _ in range(n_layers - 2):
                convs.append( GraphConv(self.hid_d, self.hid_d, bias=bias, norm=self.norm) )
            convs.append( GraphConv(self.hid_d, self.out_d, bias=bias, norm=self.norm) )
        else:
            convs.append( GraphConv(self.in_d, self.out_d, bias=bias, norm=self.norm) )
        return convs


class GraphSAGE(BaselineArch):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers=2, act=nn.ReLU(), l_act=nn.Identity(),
                 bias=True, dropout=0, aggregator='mean'):
        self.aggregator = aggregator
        super(GraphSAGE, self).__init__(in_dim, hid_dim, out_dim, n_layers, act, l_act,
                                        bias, dropout)
    
    def _create_conv_layers(self, n_layers: int, bias: bool) -> nn.ModuleList:
        convs = nn.ModuleList()

        if n_layers > 1:
            convs.append(SAGEConv(self.in_d, self.hid_d, self.aggregator, bias=bias))
            for _ in range(n_layers - 2):
                convs.append(SAGEConv(self.hid_d, self.hid_d, self.aggregator, bias=bias))
            convs.append(SAGEConv(self.hid_d, self.out_d, self.aggregator, bias=bias))
        else:
            convs.append(SAGEConv(self.in_d, self.out_d, self.aggregator, bias=bias))

        return convs

    def forward(self, graph, h):
        return super().forward(graph, h)


class GIN(BaselineArch):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers=2, act=nn.ReLU(), l_act=nn.Identity(),
                 bias=True, dropout=0, aggregator='sum', mlp_layers = 2):
        self.aggregator = aggregator
        self.apply_func = MLP
        self.mlp_layers = mlp_layers
        super(GIN, self).__init__(in_dim, hid_dim, out_dim, n_layers, act, l_act,
                                  bias, dropout)
    
    def _create_conv_layers(self, n_layers: int, bias: bool) -> nn.ModuleList:
        convs = nn.ModuleList()

        # Last actication of apply_func is always set to None because the non-linearity is applyed in
        # the forward pass of the BaselineArch class
        if n_layers > 1:
            apply_func = self.apply_func(self.in_d, self.hid_d, self.hid_d, bias=bias, act=self.act,
                                         l_act=None, n_layers=self.mlp_layers)
            convs.append(GINConv(apply_func, self.aggregator))
            for _ in range(n_layers - 2):
                apply_func = self.apply_func(self.hid_d, self.hid_d, self.hid_d, bias=bias, act=self.act,
                                             l_act=None, n_layers=self.mlp_layers)
                convs.append(GINConv(apply_func, self.aggregator))
            apply_func = self.apply_func(self.hid_d, self.hid_d, self.out_d, bias=bias, act=self.act,
                                         l_act=None, n_layers=self.mlp_layers)
            convs.append(GINConv(apply_func, self.aggregator))
        else:
            apply_func = self.apply_func(self.in_d, self.hid_d, self.out_d, bias=bias, act=self.act,
                                         l_act=None, n_layers=self.mlp_layers)
            convs.append(GINConv(apply_func, self.aggregator))

        return convs

    def forward(self, graph, h):
        # h = h.transpose(0,1)
        return super().forward(graph, h) #.transpose(1, 0)


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers=2, dropout=0., bias=True,
                 act=nn.ReLU(), l_act=nn.Identity()):
        super(MLP, self).__init__()
        self.in_d = in_dim
        self.hid_d = hid_dim
        self.out_d = out_dim
        self.act = act
        self.l_act = l_act
        self.dropout = nn.Dropout(p=dropout)

        self.lin_layers = self._create_lin_layers(n_layers, bias)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _create_lin_layers(self, n_layers: int, bias: bool) -> nn.ModuleList:
        lin_layers = nn.ModuleList()

        if n_layers > 1:
            lin_layers.append(nn.Linear(self.in_d, self.hid_d, bias=bias))
            for _ in range(n_layers - 2):
                lin_layers.append(nn.Linear(self.hid_d, self.hid_d, bias=bias))
            lin_layers.append(nn.Linear(self.hid_d, self.out_d, bias=bias))
        else:
            lin_layers.append(nn.Linear(self.in_d, self.out_d, bias=bias))

        return lin_layers

    def forward(self, h):
        for _, lin_layer in enumerate(self.lin_layers[:-1]):
            h = self.act(lin_layer(h))
            h = self.dropout(h)

        h = self.lin_layers[-1](h)
        return self.l_act(h) if self.l_act else h


class GAT(nn.Module):
    """
    Graph Attention Network Class
    """
    def __init__(self, in_dim, hid_dim, out_dim, num_heads, gat_params,
                 act=nn.ELU(), l_act=nn.Identity(), n_layers=None):
        super(GAT, self).__init__()

        if n_layers is not None:
            print('WARNING: GAT is implemeted with a fixed number of layers. The argument is ignored.')

        self.layer1 = GATConv(in_dim, hid_dim, num_heads, **gat_params)
        # Be aware that the input dimension is hid_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = GATConv(hid_dim * num_heads, out_dim, 1, **gat_params)
        self.act = act
        self.l_act = l_act

    def forward(self, graph, h):
        h = self.layer1(graph, h)
        # concatenate
        h = h.flatten(1)
        h = self.act(h)
        h = self.layer2(graph, h)
        return self.l_act(h.squeeze())
    

##########   DEPRECATED   ##########
class GCNN_2L(nn.Module):
    """
    2-layer Graph Convolutional Neural Network Class as in Kipf
    """
    def __init__(self, in_dim, hid_dim, out_dim, act=nn.ELU(), l_act=nn.Identity(),
                 norm='both', bias=True, dropout=0):
        super(GCNN_2L, self).__init__()
        self.layer1 = GraphConv(in_dim, hid_dim, bias=bias, norm=norm)
        self.layer2 = GraphConv(hid_dim, out_dim, bias=bias, norm=norm)
        self.dropout = nn.Dropout(p=dropout)
        self.act = act
        self.l_act = l_act

    def forward(self, graph, h):
        h = self.layer1(graph, h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.layer2(graph, h)
        return self.l_act(h)
####################################

# class FixedGFGNN(nn.Module):
#     """
#     Class for a Greapg Neural Network that replaces the classical normalized A by some given graph
#     fitler
#     """
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
