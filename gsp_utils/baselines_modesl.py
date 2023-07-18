import torch
import numpy as np
from src.arch import GFGCN_Spows

class NodeClassModel:
    def __init__(self, arch, S, masks, loss=torch.nn.CrossEntropyLoss(reduction='sum'),
                 device='cpu'):
        """
        NOTE: this matrix S is a dgl object, not a tensor or a np matrix.
        """
        self.arch = arch.to(device)
        self.train_mask = masks['train']
        self.val_mask = masks['val']
        self.test_mask = masks['test']
        self.loss_fn = loss
        self.S = S

    def train(self, X, labels, n_epochs, lr, wd, eval_freq=20, optim=torch.optim.Adam, verb=False):
        opt = optim(self.arch.parameters(), lr=lr, weight_decay=wd)

        losses_train, losses_val, losses_test = [np.zeros(n_epochs) for _ in range(3)]
        accs_train, accs_val, accs_test = [np.zeros(n_epochs) for _ in range(3)]
        for i in range(n_epochs):
            self.arch.train()
            opt.zero_grad()

            labels_hat = self.arch(self.S, X)
            loss = self.loss_fn(labels_hat[self.train_mask], labels[self.train_mask])
            loss.backward()
            opt.step()

            self.arch.eval()
            with torch.no_grad():
                loss_ev = self.loss_fn(labels_hat[self.val_mask], labels[self.val_mask])
                loss_test = self.loss_fn(labels_hat[self.test_mask], labels[self.test_mask])
            
            losses_train[i] = loss.detach().cpu().item()
            losses_val[i] = loss_ev.detach().cpu().item()
            losses_test[i] = loss_test.detach().cpu().item()
            accs_train[i] = self.test(X, self.S, labels, self.train_mask)
            accs_val[i] = self.test(X, self.S, labels, self.val_mask)
            accs_test[i] = self.test(X, self.S, labels, self.test_mask)

            if (i == 0 or (i+1) % eval_freq == 0) and verb:
                print(f"Epoch {i+1}/{n_epochs} - Loss Train: {losses_train[i]:.3f} - Acc Train: {accs_train[i]:.3f} - Acc Val: {accs_val[i]:.3f} - Acc Test: {accs_test[i]:.3f}", flush=True)

        losses = {'train': losses_train, 'val': losses_val, 'test': losses_test}
        accs = {'train': accs_train, 'val': accs_val, 'test': accs_test}
        return losses, accs

    def test(self, X, S, labels, mask):
        self.arch.eval()
        with torch.no_grad():
            logits = self.arch(S, X)
            logits = logits[mask]
            labels_mask = labels[mask]
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels_mask)
            return correct.item() * 1.0 / len(labels_mask)
        

class GF_NodeClassModel(NodeClassModel):
    def __init__(self, arch, S, K, masks, loss= torch.nn.CrossEntropyLoss(reduction='sum'),
                 device='cpu'):
        """
        NOTE: this matrix S is a dgl object, not a tensor or a np matrix.
        """
        super().__init__(arch, S, masks, loss, device)
        
        # Save powers of S
        N = S.shape[0]
        S_pows = torch.Tensor(torch.empty(K-1, N, N))
        S_pows[0,:,:] = S
        for k in range(1,K-1):
            S_pows[k,:,:] = S @ S_pows[k-1,:,:]
        
        self.S = S_pows if isinstance(arch, GFGCN_Spows) else S
        self.S = self.S.to(device)

    def init_optimizers(self, optim, lr, wd):
        if self.arch.bias:
            opt_W = optim([layer.W for layer in self.arch.convs] + 
                          [layer.b for layer in self.arch.convs],
                          lr=lr, weight_decay=wd)
        else:
            opt_W = optim([layer.W for layer in self.arch.convs],
                          lr=lr, weight_decay=wd)
            
        opt_h = optim([layer.h for layer in self.arch.convs],
                                      lr=lr, weight_decay=wd)
        
        return opt_W, opt_h

    def gnn_step(self, X, labels, optim, iters):
        for _ in range(iters):
            optim.zero_grad()
            labels_hat = self.arch(self.S, X)
            loss = self.loss_fn(labels_hat[self.train_mask], labels[self.train_mask])
            loss.backward()
            optim.step()

    def train(self, X, labels, n_epochs, lr, wd, eval_freq=20, optim=torch.optim.Adam, 
              epochs_h=1, epochs_W=1, clamp=False, verb=False):
        opt_W, opt_h = self.init_optimizers(optim, lr, wd)

        losses_train, losses_val, losses_test = [np.zeros(n_epochs) for _ in range(3)]
        accs_train, accs_val, accs_test = [np.zeros(n_epochs) for _ in range(3)]
        for i in range(n_epochs):
            self.arch.train()

            # Step W
            self.gnn_step(X, labels, opt_W, epochs_W)
            # opt_W.zero_grad()
            # labels_hat = self.arch(self.S, X)
            # loss = self.loss_fn(labels_hat[self.train_mask], labels[self.train_mask])
            # loss.backward()
            # opt_W.step()
            

            # Step h
            self.gnn_step(X, labels, opt_h, epochs_h)
            # opt_h.zero_grad()
            # labels_hat = self.arch(self.S, X)
            # loss = self.loss_fn(labels_hat[self.train_mask], labels[self.train_mask])
            # loss.backward()
            # opt_h.step()

            if clamp:
                self.arch.clamp_h()

            self.arch.eval()
            with torch.no_grad():

                labels_hat = self.arch(self.S, X)
                loss = self.loss_fn(labels_hat[self.train_mask], labels[self.train_mask])

                loss_ev = self.loss_fn(labels_hat[self.val_mask], labels[self.val_mask])
                loss_test = self.loss_fn(labels_hat[self.test_mask], labels[self.test_mask])
            
            losses_train[i] = loss.detach().cpu().item()
            losses_val[i] = loss_ev.detach().cpu().item()
            losses_test[i] = loss_test.detach().cpu().item()
            accs_train[i] = self.test(X, self.S, labels, self.train_mask)
            accs_val[i] = self.test(X, self.S, labels, self.val_mask)
            accs_test[i] = self.test(X, self.S, labels, self.test_mask)

            if (i == 0 or (i+1) % eval_freq == 0) and verb:
                print(f"Epoch {i+1}/{n_epochs} - Loss Train: {losses_train[i]:.3f} - Acc Train: {accs_train[i]:.3f} - Acc Val: {accs_val[i]:.3f} - Acc Test: {accs_test[i]:.3f}", flush=True)

        losses = {'train': losses_train, 'val': losses_val, 'test': losses_test}
        accs = {'train': accs_train, 'val': accs_val, 'test': accs_test}
        return losses, accs