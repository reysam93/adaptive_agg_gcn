import torch

class NodeClassModel:
    def __init__(self, arch, S, train_mask, val_mask, test_mask,
                 loss= torch.nn.CrossEntropyLoss(reduction='sum')):
        self.arch = arch
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.loss_fn = loss
        self.S = S

    def train(self, X, labels, n_epochs, lr, wd, eval_freq=20, optim=torch.optim.Adam, verb=False):
        opt = optim(self.arch.parameters(), lr=lr, weight_decay=wd)

        loss_train, loss_val, loss_test = [torch.zeros(n_epochs) for _ in range(3)]
        acc_train, acc_val, acc_test = [torch.zeros(n_epochs) for _ in range(3)]
        for i in range(n_epochs):
            self.arch.train()
            opt.zero_grad()

            labels_hat = self.arch(self.S, X)
            loss = self.loss_fn(labels_hat[self.train_mask], labels[self.train_mask])
            loss.backward()
            opt.step()

            self.arch.eval()
            with torch.no_grad():
                loss_ev = self.loss_fn(labels_hat[self.val_mask], labels[self.val_mask]).item()
                loss_test = self.loss_fn(labels_hat[self.test_mask], labels[self.test_mask]).item()
            
            loss_train[i] = loss.item()
            loss_val[i] = loss_ev
            loss_test[i] = loss_test
            acc_train[i] = self.test(X, labels, self.train_mask)
            acc_val[i] = self.test(X, labels, self.val_mask)
            acc_test[i] = self.test(X, labels, self.test_mask)

            if (i == 0 or (i+1) % eval_freq == 0) and verb:
                print(f"Epoch {i+1}/{n_epochs} - Loss Train: {loss_train[i]} - Acc Train: {acc_train[i]} - Acc Val: {acc_val[i]} - Acc Test: {acc_test[i]}", flush=True)

        losses = {'train': loss_train, 'val': loss_val, 'test': loss_test}
        accs = {'train': acc_train, 'val': acc_val, 'test': acc_test}
        return losses, accs


    def test(self, X, labels, mask):
        self.arch.eval()
        with torch.no_grad():
            logits = self.arch(self.S, X)
            logits = logits[mask]
            labels_mask = labels[mask]
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels_mask)
            return correct.item() * 1.0 / len(labels_mask)