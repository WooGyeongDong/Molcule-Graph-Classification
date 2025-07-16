import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import GCNConv, GINConv, GATConv

from ogb.utils.features import get_atom_feature_dims

full_atom_feature_dims = get_atom_feature_dims()

class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()
        
        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim+1, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:,i])

        return x_embedding


class MLP_layer(nn.Module):
    def __init__(self, hidden, hidden_out):
        super().__init__()
        self.fc1_bn = BatchNorm1d(hidden)
        self.fc1 = Linear(hidden, hidden)
        self.fc2_bn = BatchNorm1d(hidden)
        self.fc2 = Linear(hidden, hidden_out)
        
    def forward(self, x):
        x = self.fc1_bn(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2_bn(x)
        x = self.fc2(x)
        return x

class GNN_layer(nn.Module):
    def __init__(self, model, hidden, dropout):
        super().__init__()    
        self.bn = BatchNorm1d(hidden)
        if model == 'GCN':
            self.conv = GCNConv(hidden, hidden)
        elif model == 'GIN':
            self.conv = GINConv(
                Sequential(
                    Linear(hidden, hidden),
                    ReLU(),
                    Linear(hidden, hidden)
                )
            )
        elif model == 'GAT':
            self.conv = GATConv(hidden, hidden, heads=4, concat=False)
            
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = self.bn(x)
        x = self.conv(x, edge_index)
        x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        return x

class BasicGNN(nn.Module):
    def __init__(self, args, dropout=0.5):
        super().__init__()
        num_conv_layers = args.layers
        hidden = args.hidden
        self.args = args
        self.global_pool = global_add_pool
        self.dropout = dropout
        self.atom_encoder = AtomEncoder(hidden)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_conv_layers):
                self.convs.append(
                    GNN_layer(args.model, hidden, dropout))
            
        self.readout_layer = MLP_layer(hidden, args.num_classes*args.num_task)
        
    def forward(self, data, silence_node=None):
        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        x = self.atom_encoder(x)
        
        if silence_node is not None: x[silence_node] = torch.randn_like(x[silence_node])
        
        for conv in self.convs:
                x = conv(x, edge_index)        

        x = self.global_pool(x, batch)

        logit = self.readout_layer(x)
        return logit