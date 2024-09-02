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
        X_out = X @ self.W
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
    def __init__(self, in_dim, out_dim, K, bias, init_h0=1):
        super().__init__()
        self.K = K
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.b = None
        
        self.W = nn.Parameter(torch.empty((K, self.in_dim, self.out_dim)))
        torch.nn.init.kaiming_uniform_(self.W.data)
        self.W.data[0] *= init_h0

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
                 diff_layer=GFGCNLayer, init_h0=1, batch_norm=False):
        super().__init__()
        self.act = act
        self.last_act =  last_act
        self.dropout = nn.Dropout(p=dropout)
        self.n_layers = n_layers
        self.bias = bias
        
        self.convs = nn.ModuleList()

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.bn_layers = nn.ModuleList()

        self.convs.append(diff_layer(in_dim, hid_dim, K, bias, init_h0))
        
        if n_layers > 1:
            if self.batch_norm:
                self.bn_layers.append(nn.BatchNorm1d(hid_dim))

            for _ in range(n_layers - 2):
                self.convs.append(diff_layer(hid_dim, hid_dim, K, bias, init_h0))
                if self.batch_norm:
                    self.bn_layers.append(nn.BatchNorm1d(hid_dim))
            self.convs.append(diff_layer(hid_dim, out_dim, K, bias, init_h0))

    def clamp_h(self):
        with torch.no_grad():
            for layer in self.convs:
                layer.h.data = layer.h.clamp(min=0., max=1.)

    def forward(self, S, X):
        for i in range(self.n_layers - 1):
            X = self.act(self.convs[i](X, S))
            if self.batch_norm:
                X = self.bn_layers[i](X)
            X = self.dropout(X)
        X = self.convs[-1](X, S)
        return self.last_act(X)


class Dual_GFGCN(nn.Module):
    """
    This modules is composed of two different architectures (2 GFCNN). One for the GSO S, and
    the other for its transpose S.T. 
    """
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, K, alpha=.5, bias=True,
                 act=nn.ReLU(), last_act=nn.Identity(), dropout=0,
                 diff_layer=GFGCNLayer, init_h0=1, batch_norm=False):
        super().__init__()

        # GNN for S
        self.arch = GFGCN(in_dim, hid_dim, out_dim, n_layers, K, bias, act,
                          last_act, dropout, diff_layer, init_h0, batch_norm)

        # GNN for S.T
        self.arch_t = GFGCN(in_dim, hid_dim, out_dim, n_layers, K, bias, act,
                            last_act, dropout, diff_layer, init_h0, batch_norm)

        if alpha is None:
            self.alpha = nn.Parameter(torch.Tensor([0.5]))
        else:
            self.alpha = torch.Tensor([alpha])
        
        self.bias = bias

    def forward(self, S, X):
        y1 = self.arch(S, X)
        y2 = self.arch_t(S.T, X)
        return self.alpha*y1 + (1 - self.alpha)*y2


#########################################################################


####################       GNN - NodeVarian GF       ####################  
class NV_GFGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, K, N, f_type='both', groups=None, bias=True):
        super().__init__()
        self.K = K
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.b = None
        self.f_type = f_type

        self.h = nn.Parameter(torch.ones(self.K, N))

        self.W = nn.Parameter(torch.empty((self.in_dim, self.out_dim)))
        torch.nn.init.kaiming_uniform_(self.W.data)

        if bias:
            self.b = nn.Parameter(torch.empty(self.out_dim))
            torch.nn.init.constant_(self.b.data, 0.)

    def forward(self, X, S_pows):
        assert self.K-1 == S_pows.shape[0]

        H = self.h[0] * torch.eye(X.shape[0]).to(self.h.device)
        for k in range(0, self.K-1):
            if self.f_type == 'left':
                # Perform operation diag(h[k]) @ S^k
                H += self.h[k+1,:,None] * S_pows[k]
            elif self.f_type == 'right':
                # Perform operation S^k @ diag(h[k])
                H += S_pows[k] * self.h[k+1]
            else:
                # Perform operation diag(h[k]) @ S^k @ diag(h[k])
                H += self.h[k+1,:,None] * S_pows[k] * self.h[k+1]

        if self.b is not None:
            return H @ (X @ self.W) + self.b[None,:]
        else:
            return H @ (X @ self.W)


class NV_GFGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, K, N, f_type='both',
                 groups=None, bias=True, act=nn.ReLU(), last_act=nn.Identity(),
                 dropout=0):
        super().__init__() 
        self.act = act
        self.last_act =  last_act
        self.dropout = nn.Dropout(p=dropout)
        self.n_layers = n_layers
        self.bias = bias

        self.convs = self._create_conv_layers(in_dim, hid_dim, out_dim, n_layers, 
                                              K, N, f_type, groups, bias)

    def _create_conv_layers(self, in_dim, hid_dim, out_dim, n_layers,
                            K, N, f_type, groups, bias) -> nn.ModuleList:
        convs = nn.ModuleList()
        
        if n_layers > 1:
            convs.append(NV_GFGCNLayer(in_dim, hid_dim, K, N, f_type, groups, bias))
            for _ in range(n_layers - 2):
                convs.append(NV_GFGCNLayer(hid_dim, hid_dim, K, N, f_type, groups, bias))
            convs.append(NV_GFGCNLayer(hid_dim, out_dim, K, N, f_type, groups, bias))
        else:
            convs.append(NV_GFGCNLayer(in_dim, out_dim, K, N, f_type, groups, bias))

        return convs


    def forward(self, S_pows, X):
        for i in range(self.n_layers - 1):
            X = self.act(self.convs[i](X, S_pows))
            X = self.dropout(X)
        X = self.convs[-1](X, S_pows)
        return self.last_act(X)
#########################################################################


class GFGCN_SpowsLayer(GFGCNLayer):
    def forward(self, X, S_pows, norm, dev='cpu'):
        assert self.K-1 == S_pows.shape[0]

        H0 = self.h[0] * torch.eye(X.shape[0]).to(dev)
        H = (self.h[1:, None, None] * S_pows).sum(axis=0) + H0

        if norm:
            deg = H.sum(1)
            d_inv_sqr = torch.sqrt(torch.abs(torch.where(deg == 0, 0, 1/deg)))
            # Replace diagonal matrix by vectors to scale rows/columns
            H = d_inv_sqr * (H.T * d_inv_sqr).T

        if self.b is not None:
            return H @ (X @ self.W) + self.b[None,:]
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

