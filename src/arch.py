import torch.nn as nn
import torch


##########     LAYERS     ##########
class GCNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias):
        super().__init__()
        self.N = self.S.shape[1]
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.b = None
        
        self.W = nn.Parameter(torch.empty((self.in_dim, self.out_dim)))
        torch.nn.init.kaiming_uniform_(self.W.data)

        if bias:
            self.b = nn.Parameter(torch.empty(self.out_dim))
            torch.nn.init.constant_(self.b.data, 0.)

    def forward(self, X, S):
        # Nodes x Features
        Nin, Fin = X.shape
        assert Nin == self.N
        assert Fin == self.in_dim

        if self.b is not None:
            return S @ X @ self.W + self.b[None,:]
        else:
            return S @ X @ self.W
        

class GFGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, K, bias):
        super().__init__()
        self.N = self.S.shape[1]
        self.K = K
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.b = None

        self.h = nn.Parameter(torch.empty((self.K)))
        torch.nn.init.constant_(self.h.data, 1.)
        
        self.W = nn.Parameter(torch.empty((self.in_dim, self.out_dim)))
        torch.nn.init.kaiming_uniform_(self.W.data)

        if bias:
            self.b = nn.Parameter(torch.empty(self.out_dim))
            torch.nn.init.constant_(self.b.data, 0.)

    def forward(self, X, S):
        # Nodes x Features
        Nin, Fin = X.shape
        assert Nin == self.N
        assert Fin == self.in_dim

        X_out = self.h[0] * X
        Sx = X
        for k in range(1, self.K):
            Sx = S @ Sx
            X_out += self.h[k] * Sx

        if self.b is not None:
            return X_out @ self.W + self.b[None,:]
        else:
            return X_out @ self.W


##########     ARCHITECTURES     ##########
class GCNN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, bias=True,
                 nonlin=nn.ReLU, last_nonlin=nn.Identity):
        super().__init__()
        self.N = self.S.shape[0]
        self.nonlin = nonlin()
        self.last_nonlin =  last_nonlin()
        self.n_layers = n_layers
        self.convs = nn.ModuleList()

        self.convs.append(GCNNLayer(in_dim, hid_dim, bias))
        if n_layers > 1:
            for _ in range(n_layers - 2):
                self.convs.append(GCNNLayer(hid_dim, hid_dim, bias))
            self.convs.append(GCNNLayer(hid_dim, out_dim, bias))

    def forward(self, X, S):
        for i in range(self.n_layers - 1):
            X = self.nonlin(self.convs[i](X, S))
        X = self.convs[-1](X, S)
        return self.last_nonlin(X)


class GFGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, K, bias=True,
                 nonlin=nn.ReLU, last_nonlin=nn.Identity):
        super().__init__()        
        self.N = self.S.shape[0]
        self.nonlin = nonlin()
        self.last_nonlin =  last_nonlin()
        self.n_layers = n_layers
        self.convs = nn.ModuleList()

        self.convs.append(GFGCNLayer(in_dim, hid_dim, K, bias))
        if n_layers > 1:
            for _ in range(n_layers - 2):
                self.convs.append(GFGCNLayer(hid_dim, hid_dim, K, bias))
            self.convs.append(GFGCNLayer(hid_dim, out_dim, K, bias))

    def forward(self, X, S):
        for i in range(self.n_layers - 1):
            X = self.nonlin(self.convs[i](X, S))
        X = self.convs[-1](X, S)
        return self.last_nonlin(X)
            