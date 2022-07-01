import argparse
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
import torch.nn as nn

from train_eval import *
from datasets import *

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', type=bool, default=True)
parser.add_argument('--runs', type=int, default=100)
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=100)
parser.add_argument('--hidden', type=int, default=48)
parser.add_argument('--dropout1', type=float, default=0.5)
parser.add_argument('--dropout2', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--save_path', type=str, default=None)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--output_heads', type=int, default=1)
parser.add_argument('--kernel', type=int, default=5)
parser.add_argument('--model', type=str, default='NLMLP')

args = parser.parse_args()
print(args)



### gcn-based
class NLGCN(torch.nn.Module):
    def __init__(self, dataset):
        super(NLGCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes)
        self.proj = nn.Linear(dataset.num_classes, 1)
        self.kernel = 5
        self.conv1d = nn.Conv1d(dataset.num_classes, dataset.num_classes, self.kernel, padding=int((self.kernel-1)/2))
        self.conv1d_2 = nn.Conv1d(dataset.num_classes, dataset.num_classes, self.kernel, padding=int((self.kernel-1)/2))
        self.lin = nn.Linear(2*dataset.num_classes, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.proj.reset_parameters()
        self.conv1d.reset_parameters()
        self.conv1d_2.reset_parameters()
        self.lin.reset_parameters()
        

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=args.dropout1, training=self.training)
        x1 = self.conv2(x, edge_index)
        
        g_score = self.proj(x1)  # [num_nodes, 1]
        g_score_sorted, sort_idx = torch.sort(g_score, dim=0)
        _, inverse_idx = torch.sort(sort_idx, dim=0)
        
        sorted_x = g_score_sorted*x1[sort_idx].squeeze()
        sorted_x = torch.transpose(sorted_x, 0, 1).unsqueeze(0) # [1, dataset.num_classes, num_nodes]
        sorted_x = F.relu(self.conv1d(sorted_x))
        sorted_x = F.dropout(sorted_x, p=args.dropout2, training=self.training)
        sorted_x = self.conv1d_2(sorted_x)
        sorted_x = torch.transpose(sorted_x.squeeze(), 0, 1) # [num_nodes, dataset.num_classes]
        x2 = sorted_x[inverse_idx].squeeze()  # [num_nodes, dataset.num_classes]
        
        out = torch.cat([x1,x2], dim=1)
        out = self.lin(out)
        
#         g_score_cpu = g_score.cpu()
#         np.save(args.save_path, g_score_cpu.detach().numpy())
        
        return F.log_softmax(out, dim=1)
    

###gat-based
class NLGAT(torch.nn.Module):
    def __init__(self, dataset):
        super(NLGAT, self).__init__()
        self.conv1 = GATConv(
            dataset.num_features,
            args.hidden,
            heads=args.heads,
            dropout=args.dropout1)
        self.conv2 = GATConv(
            args.hidden * args.heads,
            dataset.num_classes,
            heads=args.output_heads,
            concat=False,
            dropout=args.dropout1)
        self.proj = nn.Linear(dataset.num_classes, 1)
        self.kernel = 5
        self.conv1d = nn.Conv1d(dataset.num_classes, dataset.num_classes, self.kernel, padding=int((self.kernel-1)/2))
        self.conv1d_2 = nn.Conv1d(dataset.num_classes, dataset.num_classes, self.kernel, padding=int((self.kernel-1)/2))
        self.lin = nn.Linear(2*dataset.num_classes, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.proj.reset_parameters()
        self.conv1d.reset_parameters()
        self.conv1d_2.reset_parameters()
        self.lin.reset_parameters()
        

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=args.dropout1, training=self.training)
        x1 = self.conv2(x, edge_index)

        g_score = self.proj(x1)  # [num_nodes, 1]
        g_score_sorted, sort_idx = torch.sort(g_score, dim=0)
        _, inverse_idx = torch.sort(sort_idx, dim=0)
        
        sorted_x = g_score_sorted*x1[sort_idx].squeeze()
        sorted_x = torch.transpose(sorted_x, 0, 1).unsqueeze(0) # [1, dataset.num_classes, num_nodes]
        sorted_x = F.relu(self.conv1d(sorted_x))
        sorted_x = F.dropout(sorted_x, p=args.dropout2, training=self.training)
        sorted_x = self.conv1d_2(sorted_x)
        sorted_x = torch.transpose(sorted_x.squeeze(), 0, 1) # [num_nodes, dataset.num_classes]
        x2 = sorted_x[inverse_idx].squeeze()  # [num_nodes, dataset.num_classes]
        
        out = torch.cat([x1,x2], dim=1)
        out = self.lin(out)
        
#         x_cpu = x.cpu()
#         np.save(args.save_path, x_cpu.detach().numpy()) #save embedding for visualization
        
        return F.log_softmax(out, dim=1)
    
###mlp-based    
class NLMLP(torch.nn.Module):
    def __init__(self, dataset):
        super(NLMLP, self).__init__()
        self.lin1 = nn.Linear(dataset.num_features, args.hidden)
        self.lin2 = nn.Linear(args.hidden, dataset.num_classes)
        self.proj = nn.Linear(dataset.num_classes, 1)
        self.kernel = args.kernel
        self.conv1d = nn.Conv1d(dataset.num_classes, dataset.num_classes, self.kernel, padding=int((self.kernel-1)/2))
        self.conv1d_2 = nn.Conv1d(dataset.num_classes, dataset.num_classes, self.kernel, padding=int((self.kernel-1)/2))
        self.lin = nn.Linear(2*dataset.num_classes, dataset.num_classes)


    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.proj.reset_parameters()
        self.conv1d.reset_parameters()
        self.conv1d_2.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=args.dropout1, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=args.dropout1, training=self.training)
        x1 = self.lin2(x)
        
        g_score = self.proj(x1)  # [num_nodes, 1]
        g_score_sorted, sort_idx = torch.sort(g_score, dim=0)
        _, inverse_idx = torch.sort(sort_idx, dim=0)
        
        sorted_x = g_score_sorted*x1[sort_idx].squeeze()
        sorted_x = torch.transpose(sorted_x, 0, 1).unsqueeze(0) # [1, dataset.num_classes, num_nodes]
        sorted_x = F.relu(self.conv1d(sorted_x))
        sorted_x = F.dropout(sorted_x, p=args.dropout2, training=self.training)
        sorted_x = self.conv1d_2(sorted_x)
        sorted_x = torch.transpose(sorted_x.squeeze(), 0, 1) # [num_nodes, dataset.num_classes]
        x2 = sorted_x[inverse_idx].squeeze()  # [num_nodes, dataset.num_classes]
        
        out = torch.cat([x1,x2], dim=1)
        out = self.lin(out)
        
#         g_score_cpu = g_score.cpu()
#         np.save(args.save_path, g_score_cpu.detach().numpy())
        

        return F.log_softmax(out, dim=1)


if args.dataset == "Cora" or args.dataset == "CiteSeer" or args.dataset == "PubMed":
    dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
    permute_masks = random_planetoid_splits if args.random_splits else None
    print("Data:", dataset)
    if args.model=="NLMLP":
        Net = NLMLP
    elif args.model=="NLGCN":
        Net = NLGCN
    elif args.model=="NLGAT":
        Net = NLGAT
    else:
        print("Please choose a correct model!")
    run(dataset, Net(dataset), args.runs, args.epochs, args.lr, args.weight_decay, args.early_stopping, permute_masks, lcc=False)
elif args.dataset == "chameleon" or args.dataset == "squirrel" or args.dataset == "film" or args.dataset == "cornell" or args.dataset == "texas" or args.dataset == "wisconsin":
    dataset = get_disassortative_dataset(args.dataset)
    permute_masks = random_disassortative_splits
    print("Data:", dataset)
    if args.model=="NLMLP":
        Net = NLMLP
    elif args.model=="NLGCN":
        Net = NLGCN
    elif args.model=="NLGAT":
        Net = NLGAT
    else:
        print("Please choose a correct model!")
    run_disassortative_dataset(dataset, Net(dataset), args.runs, args.epochs, args.lr, args.weight_decay, args.early_stopping, permute_masks, lcc=False)

