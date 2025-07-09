import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool


class ToxGNN(nn.Module):
    def __init__(self, n_node_feats: int, n_edge_feats: int, n_tasks: int):
        super().__init__()

        def gin_layer(in_dim, out_dim):
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(out_dim, out_dim)
            )
            return GINEConv(mlp, edge_dim=n_edge_feats)

        self.convs = nn.ModuleList([
            gin_layer(n_node_feats, 64),
            gin_layer(64, 128),
            gin_layer(128, 256),
        ])

        self.pool = global_add_pool
        self.head = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_tasks)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        edge_attr = edge_attr.float()
        
        if torch.isnan(x).any():
            print("Warning: NaN detected in node features in model forward pass")
            x = torch.zeros_like(x)
        if torch.isnan(edge_attr).any():
            print("Warning: NaN detected in edge features in model forward pass")
            edge_attr = torch.zeros_like(edge_attr)
        
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index, edge_attr))
            
            if torch.isnan(x).any():
                print(f"Warning: NaN detected after conv layer {i}")
                x = torch.zeros_like(x)
        
        x = self.pool(x, batch)
        
        if torch.isnan(x).any():
            print("Warning: NaN detected after pooling")
            x = torch.zeros_like(x)
        
        output = self.head(x)
        
        if torch.isnan(output).any():
            print("Warning: NaN detected in model output")
            output = torch.zeros_like(output)
        
        return output
