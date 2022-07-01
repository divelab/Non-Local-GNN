import os.path as osp
import torch

from torch_geometric.datasets import Planetoid, Coauthor, Amazon
import torch_geometric.transforms as T


def get_planetoid_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = Planetoid(path, name)

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    return dataset

def get_disassortative_dataset(name):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name, 'data.pt')
    data = torch.load(path)
    data.num_classes = 5
    
    return data
