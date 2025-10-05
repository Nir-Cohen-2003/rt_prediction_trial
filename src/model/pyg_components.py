import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from typing import Literal

class TransformerPool(nn.Module):
    def __init__(self, in_channels, num_heads=4, dim_feedforward=128, dropout_rate=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x, edge_index, batch, edge_attr=None):
        attn_mask = batch.unsqueeze(0) != batch.unsqueeze(1)
        x = x.unsqueeze(0)
        x = self.encoder(x, mask=attn_mask)
        x = x.squeeze(0)
        is_first = torch.cat([torch.tensor([True], device=x.device), batch[1:] != batch[:-1]])
        return x[is_first]

class SAGPool(nn.Module):
    def __init__(self, in_channels, ratio=0.5):
        super().__init__()
        self.pool = gnn.SAGPooling(in_channels, ratio)

    def forward(self, x, edge_index, batch, edge_attr=None):
        # pass edge_attr into the pool in the correct argument position
        x, edge_index, edge_attr, batch, _, _ = self.pool(x, edge_index, edge_attr, batch)
        return gnn.global_mean_pool(x, batch)

class TopKPool(nn.Module):
    def __init__(self, in_channels, ratio=0.5):
        super().__init__()
        self.pool = gnn.TopKPooling(in_channels, ratio)

    def forward(self, x, edge_index, batch, edge_attr=None):
        # pass edge_attr into the pool in the correct argument position
        x, edge_index, edge_attr, batch, _, _ = self.pool(x, edge_index, edge_attr, batch)
        return gnn.global_mean_pool(x, batch)

def get_activation(name: Literal["relu", "silu", "gelu"]) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    elif name == "silu":
        return nn.SiLU()
    elif name == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unknown activation function: {name}, supported: 'relu', 'silu', 'gelu'")