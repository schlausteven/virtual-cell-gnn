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
                nn.Dropout(0.1),  # Add dropout to prevent overfitting
                nn.Linear(out_dim, out_dim)
            )
            # 👇 tell GINEConv how wide the edge feature vectors are
            return GINEConv(mlp, edge_dim=n_edge_feats)

        self.convs = nn.ModuleList([
            gin_layer(n_node_feats, 64),
            gin_layer(64, 128),
            gin_layer(128, 256),
        ])

        self.pool = global_add_pool
        self.head = nn.Sequential(
            nn.BatchNorm1d(256),        # ⇐ new layer spreads activations
            nn.ReLU(),
            nn.Dropout(0.3),            # Increased dropout
            nn.Linear(256, 128),        # Add intermediate layer
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_tasks)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        edge_attr = edge_attr.float()
        
        # Check for NaN in inputs
        if torch.isnan(x).any():
            print("Warning: NaN detected in node features in model forward pass")
            x = torch.zeros_like(x)
        if torch.isnan(edge_attr).any():
            print("Warning: NaN detected in edge features in model forward pass")
            edge_attr = torch.zeros_like(edge_attr)
        
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index, edge_attr))
            
            # Check for NaN after each convolution
            if torch.isnan(x).any():
                print(f"Warning: NaN detected after conv layer {i}")
                x = torch.zeros_like(x)
        
        x = self.pool(x, batch)
        
        # Check for NaN after pooling
        if torch.isnan(x).any():
            print("Warning: NaN detected after pooling")
            x = torch.zeros_like(x)
        
        output = self.head(x)
        
        # Check for NaN in final output
        if torch.isnan(output).any():
            print("Warning: NaN detected in model output")
            output = torch.zeros_like(output)
        
        return output
