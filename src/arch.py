import torch.nn as nn
import torch


####################     GNN - FIXED GSO (or GF)     ####################
class GCNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.b = None
        
        self.W = nn.Parameter(torch.empty((self.in_dim, self.out_dim)))
        torch.nn.init.kaiming_uniform_(self.W.data)

        if bias:
            self.b = nn.Parameter(torch.empty(self.out_dim))
            torch.nn.init.constant_(self.b.data, 0.)

    def forward(self, X, S): 
        X_out = X  @ self.W
        if S.is_sparse:
            X_out = torch.sparse.mm(S, X_out)
        else:
            X_out = S @ X_out

        if self.b is not None:
            return X_out + self.b[None,:]
        else:
            return X_out


class GCNN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, bias=True,
                 act=nn.ReLU(), last_act=nn.Identity(), dropout=0,
                 diff_layer=GCNNLayer):
        super().__init__()
        self.act = act
        self.last_act =  last_act
        self.dropout = nn.Dropout(p=dropout)
        self.n_layers = n_layers
        self.bias = bias
        self.convs = nn.ModuleList()            

        self.convs.append(diff_layer(in_dim, hid_dim, bias))
        if n_layers > 1:
            for _ in range(n_layers - 2):
                self.convs.append(diff_layer(hid_dim, hid_dim, bias))
            self.convs.append(diff_layer(hid_dim, out_dim, bias))

    def forward(self, S, X):
        for i in range(self.n_layers - 1):
            X = self.act(self.convs[i](X, S))
            X = self.dropout(X)
        X = self.convs[-1](X, S)
        return self.last_act(X)

#########################################################################


####################       GNN - LEARNABLE GF       ####################
class GFGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, K, bias, init_h0=1):
        super().__init__()
        self.K = K
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.b = None

        self.h = nn.Parameter(torch.empty((self.K)))
        torch.nn.init.constant_(self.h.data, 1.)
        self.h.data[0] = init_h0

        self.W = nn.Parameter(torch.empty((self.in_dim, self.out_dim)))
        torch.nn.init.kaiming_uniform_(self.W.data)

        if bias:
            self.b = nn.Parameter(torch.empty(self.out_dim))
            torch.nn.init.constant_(self.b.data, 0.)

    def forward(self, X, S):
        X = X @ self.W
        X_out = self.h[0] * X
        Sx = X
        for k in range(1, self.K):
            Sx = S @ Sx
            X_out += self.h[k] * Sx

        if self.b is not None:
            return X_out + self.b[None,:]
        else:
            return X_out
        

class GFGCN_noh_Layer(nn.Module):
    def __init__(self, in_dim, out_dim, K, bias):
        super().__init__()
        self.K = K
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.b = None
        
        self.W = nn.Parameter(torch.empty((K, self.in_dim, self.out_dim)))
        torch.nn.init.kaiming_uniform_(self.W.data)

        if bias:
            self.b = nn.Parameter(torch.empty(self.out_dim))
            torch.nn.init.constant_(self.b.data, 0.)

    def forward(self, X, S):
        X_out = X @ self.W[0,:,:]
        Sx = X
        for k in range(1, self.K):
            Sx = S @ Sx
            X_out += Sx @ self.W[k,:,:]

        if self.b is not None:
            return X_out + self.b[None,:]
        else:
            return X_out


class GFGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, K, bias=True,
                 act=nn.ReLU(), last_act=nn.Identity(), dropout=0,
                 diff_layer=GFGCNLayer, init_h0=1):
        super().__init__()        
        self.act = act
        self.last_act =  last_act
        self.dropout = nn.Dropout(p=dropout)
        self.n_layers = n_layers
        self.bias = bias
        self.convs = nn.ModuleList()

        self.convs.append(diff_layer(in_dim, hid_dim, K, bias, init_h0))
        if n_layers > 1:
            for _ in range(n_layers - 2):
                self.convs.append(diff_layer(hid_dim, hid_dim, K, bias, init_h0))
            self.convs.append(diff_layer(hid_dim, out_dim, K, bias, init_h0))

    def clamp_h(self):
        with torch.no_grad():
            for layer in self.convs:
                layer.h.data = layer.h.clamp(min=0., max=1.)

    def forward(self, S, X):
        for i in range(self.n_layers - 1):
            X = self.act(self.convs[i](X, S))
            X = self.dropout(X)
        X = self.convs[-1](X, S)
        return self.last_act(X)

    #########################################################################


class GFGCN_SpowsLayer(GFGCNLayer):
    def forward(self, X, S_pows, norm, dev='cpu'):
        assert self.K-1 == S_pows.shape[0]

        H = self.h[0] * torch.eye(X.shape[0]).to(dev)
        for k in range(0, self.K-1):
            H += self.h[k+1] * S_pows[k,:,:]

        if norm:
            d_inv_sqr = torch.sqrt(torch.abs(1/H.sum(1)))
            # Replace diagonal matrix by vectors to scale rows/columns
            H = d_inv_sqr * (H.T * d_inv_sqr).T
            # H = D_inv_sqr @ H @ D_inv_sqr


        if self.b is not None:
            return H @ (X @ self.W) + self.b[None,:]      # NxNxFo + NxFixFo
        else:
            return H @ (X @ self.W)


class GFGCN_Spows(GFGCN):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, K, norm=True, bias=True,
                 act=nn.ReLU(), last_act=nn.Identity(), dropout=0, dev='cpu'):
        super().__init__(in_dim, hid_dim, out_dim, n_layers, K, bias, act, last_act, dropout,
                         diff_layer=GFGCN_SpowsLayer) 
        self.norm = norm
        self.dev = dev

    def forward(self, S_pows, X):
        for i in range(self.n_layers - 1):
            X = self.act(self.convs[i](X, S_pows, self.norm, dev=self.dev))
            X = self.dropout(X)
        X = self.convs[-1](X, S_pows, self.norm, dev=self.dev)
        return self.last_act(X)

