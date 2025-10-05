import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from typing import Literal
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import softmax as pyg_softmax

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

class AttentiveFPReadout(nn.Module):
    """
    AttentiveFP Graph Readout with GRU-based attention.
    
    Based on "Pushing the Boundaries of Molecular Representation for Drug Discovery 
    with the Graph Attention Mechanism" (Xiong et al., 2020)
    
    Args:
        feat_size: Size of node features
        num_timesteps: Number of GRU timesteps for attention
        dropout: Dropout rate
    """
    
    def __init__(self, feat_size, num_timesteps=2, dropout=0.0):
        super(AttentiveFPReadout, self).__init__()
        
        self.feat_size = feat_size
        self.num_timesteps = num_timesteps
        
        # GRU for iterative readout
        self.gru = nn.GRUCell(feat_size, feat_size)
        
        # Attention scoring
        self.attend = nn.Linear(feat_size, feat_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, node_feats, batch):
        """
        Args:
            node_feats: Node features [num_nodes, feat_size]
            batch: Batch assignment [num_nodes]
        
        Returns:
            Graph-level features [batch_size, feat_size]
        """
        batch_size = int(batch.max().item()) + 1
        
        # Initialize with mean pooling
        graph_feats = global_mean_pool(node_feats, batch)
        
        # Iterative attention-based readout
        for _ in range(self.num_timesteps):
            # Compute attention scores
            # Expand graph_feats to match node dimension
            graph_feats_expanded = graph_feats[batch]  # [num_nodes, feat_size]
            
            # Attention mechanism
            attended_node = self.attend(node_feats)  # [num_nodes, feat_size]
            attention_scores = (attended_node * graph_feats_expanded).sum(dim=-1)  # [num_nodes]
            
            # Softmax over nodes in each graph using PyG's softmax
            attention_weights = pyg_softmax(attention_scores, batch, dim=0)  # [num_nodes]
            
            # Weighted sum of node features
            weighted_feats = node_feats * attention_weights.unsqueeze(-1)  # [num_nodes, feat_size]
            context = global_add_pool(weighted_feats, batch)  # [batch_size, feat_size]
            
            # Update graph representation with GRU
            context = self.dropout(context)
            graph_feats = self.gru(context, graph_feats)
        
        return graph_feats