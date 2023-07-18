import numpy as np
import torch.nn as nn
import torch

import arch


class MinimaxModel:
    def __init__(self, S0, iters, lr, wd, model_params, eval_freq, 
                 loss=nn.CrossEntropyLoss(), dev='cpu'):
        # NOTE: use different lr for the optimizers??
        # TODO: Check if I need to detach S0??
        self.iters_out = iters['out']
        self.iters_W = iters['W']
        self.iters_S = iters['S']
        self.lr = lr

        self.loss_fn = loss
        self.eval_freq = eval_freq

        self.S = nn.Parameter(S0)
        self.build_model(lr, wd, model_params)
        self.arch.to(dev)

    def evaluate_clas(self, features, labels, mask):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(features)
            logits = logits[mask]
            labels = labels[mask]
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
            return correct.item() * 1.0 / len(labels)

    def loss_fn_S(self, S0, Y_hat, Y, lamb, beta, alpha):
        loss = -1*self.loss_fn(Y_hat, Y) + lamb*torch.sum(self.S)
        loss += beta*(1 - alpha)*torch.linalg.norm(self.S - S0, 'fro')
        return loss

    def step_S(self, S0, X, Y, regs, train_idx, val_idx=[], test_idx=[], 
               S_true=None, debug=False):
        train_loss, val_loss, test_loss = [np.zeros(self.iters_S) for _ in range(3)]
        errs_S = np.zeros(self.n_iters_S)
        norm_S = torch.linalg.norm(S_true)

        lamb = regs['lambda']*self.lr
        beta = regs['beta']
        beta_sc *= self.lr*alpha
        alpha = regs['alpha']

        for i in range(self.iters_S):
            self.arch.train()
            self.opt_S.zero_grad()

            # Gradient Step
            Y_hat = self.arch(X, self.S)
            loss = self.loss_fn_S(Y_hat[train_idx], Y[train_idx], lamb, beta, alpha)
            loss.backward()
            self.opt_S.step()

            # Proximal for the distance to S0
            idxs_greater = torch.where(S - S0 > beta)
            idxs_lower = torch.where(S - S0 < -beta)
            S_prox = S0.clone()
            S_prox[idxs_greater] = S[idxs_greater] - beta
            S_prox[idxs_lower] = S[idxs_lower] + beta
            S = S_prox

            # Projection onto \mathcal{A}
            S = torch.clamp(S, min=0., max=1.)
            S = (S + S.T) / 2

            ARE YOU UPDATING S??

            # Compute loss on training/validation/test for debug
            with torch.no_grad():
                if S_true is not None:
                    errs_S[i] = torch.linalg.norm(S - S_true) / norm_S

                loss_train_val = self.loss_fn(Y_hat[train_idx], Y[train_idx]).item()
                loss_ev_val = self.loss_fn(Y_hat[val_idx], Y[val_idx]).item()
                loss_test_val = self.loss_fn(Y_hat[test_idx], Y[test_idx]).item()
            
            train_loss[i] = loss.item()
            val_loss[i] = loss_ev_val
            test_loss[i] = loss_test_val

        return errs_S, {'train': train_loss, 'val': val_loss, 'test': test_loss}

    def GNN_step(self, X, Y, iters, optim, train_idx, val_idx, test_idx,
                 label, verbose=False):
        train_loss, val_loss, test_loss = [np.zeros(iters) for _ in range(3)]
        for i in range(iters):
            self.arch.train()
            optim.zero_grad()

            Y_hat = self.arch(X, self.S)
            loss = self.loss_fn(Y_hat[train_idx], Y[train_idx])
            loss.backward()
            optim.step()
            
            # NOTE: maybe useful for classification 
            # ev_train[i] = self.evaluate_clas(X, Y, train_idx)
            # ev_val[i] = self.evaluate_clas(X, Y, val_idx)
            # ev_test[i] = self.evaluate_clas(X, Y, test_idx)

            # Compute loss on training/validation/test for debug
            # TODO: Early stopping or Keep Best arch
            self.model.eval()
            with torch.no_grad():
                # TODO: estimate Y_hat with new values for val?
                loss_ev_val = self.loss_fn(Y_hat[val_idx], Y[val_idx]).item()
                loss_test_val = self.loss_fn(Y_hat[test_idx], Y[test_idx]).item()
            

            # KEEP BEST VAL ERROR
            # if loss_val < self.best_val_loss:
            # self.best_val_loss = loss_val
            # self.best_graph = normalized_adj.detach()
            # self.weights = deepcopy(self.model.state_dict())
            # if args.debug:
            #     print(f'\t=== saving current graph/gcn, best_val_loss: %s' % self.best_val_loss.item())


            train_loss[i] = loss.item()
            val_loss[i] = loss_ev_val
            test_loss[i] = loss_test_val

            if (i == 0 or (i+1) % self.eval_freq == 0) and verbose:
                print(f"\tEpoch ({label}) {i+1}/{iters} - Loss: {train_loss[i]:.3f} - Val Loss: {val_loss[i]:.3f} - Test Loss: {test_loss[i]:.3f}")

        return {'train': train_loss, 'val': val_loss, 'test': test_loss}



class MinimaxGCNNModel(MinimaxModel):
    def build_model(self, lr, wd, model_params):
        self.arch = arch.GCNN(**model_params)
        
        if model_params['bias']:
            self.opt_W = torch.optim.Adam(
                [layer.W for layer in self.model.convs] + 
                [layer.b for layer in self.model.convs],
                lr=lr, weight_decay=wd)
        else:
            self.opt_W = torch.optim.Adam(
                [layer.W for layer in self.model.convs],
                lr=lr, weight_decay=wd)
        
        self.opt_S = torch.optim.SGD([self.S], lr=lr)

    def normalize_S(self):
        return


    def fit(self, S0, X, Y, regs, train_idx=[], val_idx=[], test_idx=[],
            S_true=None, verb_out=False, verb_W=False, verb_S=False, verb_H=False):
        for i in range(self.iters_out):
            # Step S
            err_S, loss_S = self.step_S(S0, X, Y, regs, train_idx, val_idx,
                                        test_idx, S_true, verb_S)
            # TODO: Normalize S!
            # In https://github.com/DSE-MSU/DeepRobust/blob/master/deeprobust/graph/defense/prognn.py#L275
            # they normalize S first and after updating S (after they do it in eval mode)
            
            # W step
            loss_W = self.GNN_step(X, Y, self.iters_W, self.opt_W, train_idx, 
                                   val_idx, test_idx, 'W', verb_W)

            if verb_out:
                l_test = loss_W['test'][i-1]
                acc = self.evaluate_clas(X, Y, test_idx)
                print(f"Iteration {i+1} DONE - Loss Test: {l_test:.3f} Acc Test: {acc} - Err S: {err_S:.3f}")


class MinimaxGFGNNModel(MinimaxModel):
    def __init__(self, S0, iters, lr, wd, model_params, eval_freq, 
                 loss=nn.CrossEntropyLoss(), dev='cpu'):
        super().__init__(S0, iters, lr, wd, model_params, eval_freq, 
                 loss, dev)
        
        self.iters_h = iters['h']

    def fit(self, S0, X, Y, regs, train_idx=[], val_idx=[], test_idx=[],
            S_true=None, verb_out=False, verb_W=False, verb_S=False, verb_H=False):
        for i in range(self.iters_out):
            # Step S
            err_S, loss_S = self.step_S(S0, X, Y, regs, train_idx, val_idx,
                                        test_idx, S_true, verb_S)
            
            # W step
            loss_W = self.GNN_step(X, Y, self.iters_W, self.opt_W, train_idx, 
                                   val_idx, test_idx, 'W', verb_W)

            # Step h            
            loss_h = self.GNN_step(X, Y, self.iters_h, self.opt_h, train_idx,
                                   val_idx, test_idx, 'h', verb_W)

            if verb_out:
                l_test = loss_h['test'][i-1]
                acc = self.evaluate_clas(X, Y, test_idx)
                print(f"Iteration {i+1} DONE - Loss Test: {l_test:.3f} Acc Test: {acc} - Err S: {err_S:.3f}")
        
    def build_model(self, lr, wd, model_params):
        self.arch = arch.GFGCN(**model_params)
        
        if model_params['bias']:
            self.opt_W = torch.optim.Adam(
                [layer.W for layer in self.model.convs] + 
                [layer.b for layer in self.model.convs],
                lr=lr, weight_decay=wd)
        else:
            self.opt_W = torch.optim.Adam(
                [layer.W for layer in self.model.convs],
                lr=lr, weight_decay=wd)
        
        self.opt_h = torch.optim.Adam([layer.h for layer in self.model.convs],
                                      lr=lr, weight_decay=wd)
        
        self.opt_S = torch.optim.SGD([self.S], lr=lr)