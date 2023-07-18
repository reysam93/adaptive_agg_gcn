import dgl
import torch
import numpy as np


# DATA RELATED
def get_data_dgl(dataset_name, verb=False, dev='cpu'):
    dataset = getattr(dgl.data, dataset_name)(verbose=False)

    g = dataset[0]

    # get graph and node feature
    S = g.adj().to_dense().numpy()
    feat = g.ndata['feat'].to(dev)

    # get labels
    label = g.ndata['label'].to(dev)
    n_class = dataset.num_classes
    # get data split
    masks = {}
    masks['train'] = g.ndata['train_mask'].to(dev)
    masks['val'] = g.ndata['val_mask'].to(dev)
    masks['test'] = g.ndata['test_mask'].to(dev)

    if verb:
        N = S.shape[0]
        print('Dataset:', dataset_name)
        print(f'Number of nodes: {S.shape[0]}')
        print(f'Number of features: {feat.shape[1]}')
        print(f'Shape of signals: {feat.shape}')
        print(f'Number of classes: {n_class}')
        print(f'Norm of A: {np.linalg.norm(S, "fro")}')
        print(f'Max value of A: {np.max(S)}')
        print(f'Proportion of validation data: {torch.sum(masks["val"] == True).item()/N:.2f}')
        print(f'Proportion of test data: {torch.sum(masks["test"] == True).item()/N:.2f}')


    return S, feat, label, n_class, masks


def normalize_feats(X):
    rowsum = X.sum(1)
    r_inv = torch.pow(rowsum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    return r_mat_inv @ X
