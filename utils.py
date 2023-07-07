import dgl
import torch


# DATA RELATED
def get_data_dgl(dataset_name, verb=False, dev='cpu'):
    dataset = getattr(dgl.data, dataset_name)(verbose=False)

    g = dataset[0]

    # get graph and node feature
    S = g.adj().to_dense().numpy()
    feat = g.ndata['feat'].to(dev)

    # get labels
    label = g.ndata['label'].to(dev)

    # get data split
    train_mask = g.ndata['train_mask'].to(dev)
    val_mask = g.ndata['val_mask'].to(dev)
    test_mask = g.ndata['test_mask'].to(dev)

    if verb:
        N = S.shape[0]
        print('Dataset:', dataset_name)
        print(f'Number of nodes: {S.shape[0]}')
        print(f'Number of features: {feat.shape[1]}')
        print(f'Shape of signals: {feat.shape}')
        print(f'Number of classes: {dataset.num_classes}')
        print(f'Norm of A: {torch.linalg.norm(S, "fro")}')
        print(f'Max value of A: {torch.max(S)}')
        print(f'Proportion of validation data: {torch.sum(val_mask == True).item()/N:.2f}')
        print(f'Proportion of test data: {torch.sum(test_mask == True).item()/N:.2f}')


    return S, feat, label, train_mask, val_mask, test_mask