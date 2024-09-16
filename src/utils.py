import dgl
import torch
import numpy as np
from pandas import DataFrame
from IPython.display import display

# DATA RELATED
def get_data_dgl(dataset_name, verb=False, dev='cpu', idx=0):
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
        masks[lab] = mask[:,idx] if len(mask.shape) > 1 else mask

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

def summary_table(accs, datasets, exps):
    mean_accs = accs.mean(axis=2)
    cols_name = []
    for dataset_name in datasets:
        graph = getattr(dgl.data, dataset_name)(verbose=False)[0]
        edge_hom = dgl.edge_homophily(graph, graph.ndata['label'])
        cols_name.append(f'{dataset_name} ({edge_hom:.2f})')

    index_name = [exp['leg'] for exp in exps]

    return DataFrame(mean_accs, columns=cols_name, index=index_name)

def display_data(exps_leg, datasets, err, time, metric_label='Err'):
    for i, dataset in enumerate(datasets):
        err_i = err[:,i,:]
        time_i = time[:,i]

        data = {
            'Exp': exps_leg,
            f'Mean {metric_label}': err_i.mean(axis=1),
            f'Median {metric_label}': np.median(err_i, axis=1),
            'Mean Std': err_i.std(axis=1),
            'time': time_i.mean(axis=1)
        }
        df = DataFrame(data)
        display(df)

def display_all_data(exps_leg, datasets, err, agg='mean'):
    assert len(exps_leg) == err.shape[0]
    data = {'Exp': exps_leg}
    for i, dataset in enumerate(datasets):
        graph = getattr(dgl.data, dataset)(verbose=False)[0]
        edge_hom = dgl.edge_homophily(graph, graph.ndata['label'])        
        name = dataset.replace('Dataset', '')
        key = f'{name} ({edge_hom:.2f})'

        agg_err_i = getattr(np, agg)(err[:,i,:], axis=1)
        std_err_i = np.std(err[:,i,:], axis=1)

        data[key] = []
        for j in range(len(exps_leg)):
            data[key].append( f'{agg_err_i[j]:.4} \u00B1 {std_err_i[j]:.4}' )

    df = DataFrame(data)
    display(df)

