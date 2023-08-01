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
    mask_labels = ['train', 'val', 'test']
    for lab in mask_labels:
        mask = g.ndata[lab + '_mask'].to(dev)
        # Select first data splid if more than one is available
        masks[lab] = mask[:,0] if len(mask.shape) > 1 else mask

    if verb:
        N = S.shape[0]

        node_hom = dgl.node_homophily(g, g.ndata['label'])
        edge_hom = dgl.edge_homophily(g, g.ndata['label'])

        print('Dataset:', dataset_name)
        print(f'Number of nodes: {S.shape[0]}')
        print(f'Number of features: {feat.shape[1]}')
        print(f'Shape of signals: {feat.shape}')
        print(f'Number of classes: {n_class}')
        print(f'Norm of A: {np.linalg.norm(S, "fro")}')
        print(f'Max value of A: {np.max(S)}')
        print(f'Proportion of validation data: {torch.sum(masks["val"] == True).item()/N:.2f}')
        print(f'Proportion of test data: {torch.sum(masks["test"] == True).item()/N:.2f}')
        print(f'Node homophily: {node_hom:.2f}')
        print(f'Edge homophily: {edge_hom:.2f}')

    return S, feat, label, n_class, masks


def normalize_feats(X):
    rowsum = X.sum(1)
    r_inv = torch.pow(rowsum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    return r_mat_inv @ X
