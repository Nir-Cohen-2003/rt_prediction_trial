"""
Generic PyTorch Geometric model builder for various GNN architectures.

Supports GCN, GIN, and Transformer-based convolutions with flexible pooling options.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from ..config import ModelConfig
from .pyg_components import TransformerPool, SAGPool, TopKPool


class GenericPyGModel(nn.Module):
    """
    Generic PyTorch Geometric model supporting multiple GNN architectures.
    
    Args:
        node_in_dim: Input node feature dimension
        edge_in_dim: Input edge feature dimension
        hid_dim: Hidden dimension
        num_layers: Number of GNN layers
        gnn_type: Type of GNN layer ('gcn', 'gin', 'transformer')
        dropout: Dropout rate
        pool_type: Type of pooling ('mean', 'sum', 'max', 'transformer', 'sag', 'topk')
        pool_ratio: Ratio for hierarchical pooling (SAG, TopK)
        pool_num_heads: Number of heads for transformer pooling
        pool_dim_feedforward: Feedforward dimension for transformer pooling
        num_heads: Number of attention heads for TransformerConv
        edge_dim: Edge feature dimension for models that support it
        ffn_hidden_dim: Hidden dimension for output MLP
        ffn_num_layers: Number of layers in output MLP
    """
    
    def __init__(self,
                 node_in_dim,
                 edge_in_dim,
                 hid_dim,
                 num_layers,
                 gnn_type='gcn',
                 dropout=0.0,
                 pool_type='mean',
                 pool_ratio=0.5,
                 pool_num_heads=4,
                 pool_dim_feedforward=128,
                 num_heads=4,
                 edge_dim=None,
                 ffn_hidden_dim=300,
                 ffn_num_layers=2):
        super(GenericPyGModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type.lower()
        self.pool_type = pool_type
        
        # Initial encoders
        self.node_encoder = nn.Linear(node_in_dim, hid_dim)
        self.edge_encoder = nn.Linear(edge_in_dim, hid_dim) if edge_dim else None
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for layer in range(num_layers):
            if self.gnn_type == 'gcn':
                conv = gnn.GCNConv(hid_dim, hid_dim)
            
            elif self.gnn_type == 'gin':
                # GIN uses an MLP for the update function
                mlp = nn.Sequential(
                    nn.Linear(hid_dim, hid_dim),
                    nn.BatchNorm1d(hid_dim),
                    nn.ReLU(),
                    nn.Linear(hid_dim, hid_dim)
                )
                conv = gnn.GINConv(mlp, train_eps=True)
            
            elif self.gnn_type == 'transformer':
                # TransformerConv supports edge features
                conv = gnn.TransformerConv(
                    in_channels=hid_dim,
                    out_channels=hid_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=hid_dim if edge_dim else None,
                    beta=True  # Gated attention
                )
            
            else:
                raise ValueError(f"Unknown gnn_type: {self.gnn_type}")
            
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(hid_dim))
        
        # Pooling/Readout layer
        if pool_type == 'mean':
            self.readout = lambda x, batch, edge_index, edge_attr: global_mean_pool(x, batch)
        elif pool_type == 'sum':
            self.readout = lambda x, batch, edge_index, edge_attr: global_add_pool(x, batch)
        elif pool_type == 'max':
            self.readout = lambda x, batch, edge_index, edge_attr: global_max_pool(x, batch)
        elif pool_type == 'transformer':
            self.readout = TransformerPool(
                in_channels=hid_dim,
                num_heads=pool_num_heads,
                dim_feedforward=pool_dim_feedforward,
                dropout_rate=dropout
            )
        elif pool_type == 'sag':
            self.readout = SAGPool(in_channels=hid_dim, ratio=pool_ratio)
        elif pool_type == 'topk':
            self.readout = TopKPool(in_channels=hid_dim, ratio=pool_ratio)
        else:
            raise ValueError(f"Unknown pool_type: {pool_type}")
        
        # Output MLP
        mlp_layers = []
        for i in range(ffn_num_layers):
            if i == 0:
                mlp_layers.append(nn.Linear(hid_dim, ffn_hidden_dim))
            else:
                mlp_layers.append(nn.Linear(ffn_hidden_dim, ffn_hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
        mlp_layers.append(nn.Linear(ffn_hidden_dim, 1))
        
        self.out_mlp = nn.Sequential(*mlp_layers)
    
    def forward(self, data):
        """
        Args:
            data: PyG Data object with x, edge_index, edge_attr, batch
        
        Returns:
            Predicted retention times [batch_size, 1]
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Initial encoding
        x = self.node_encoder(x)
        edge_attr_encoded = self.edge_encoder(edge_attr) if self.edge_encoder else None
        
        # Message passing with residual connections
        for i, conv in enumerate(self.convs):
            x_prev = x
            
            # Apply GNN layer
            if self.gnn_type == 'gcn':
                x = conv(x, edge_index)
            elif self.gnn_type == 'gin':
                x = conv(x, edge_index)
            elif self.gnn_type == 'transformer':
                x = conv(x, edge_index, edge_attr_encoded)
            
            # Batch normalization
            x = self.batch_norms[i](x)
            
            # Activation
            x = F.relu(x)
            
            # Dropout
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection (skip first layer to avoid dimension mismatch)
            if i > 0:
                x = x + x_prev
        
        # Graph-level readout
        if self.pool_type in ['mean', 'sum', 'max']:
            graph_feat = self.readout(x, batch, edge_index, edge_attr_encoded)
        else:
            graph_feat = self.readout(x, edge_index, batch, edge_attr_encoded)
        
        # Final prediction
        out = self.out_mlp(graph_feat)
        
        return out


def build_pyg_model(model_config: ModelConfig) -> nn.Module:
    """
    Build a PyTorch Geometric model from configuration.
    
    Args:
        model_config: Model configuration
    
    Returns:
        GenericPyGModel instance
    """
    pyg_cfg = model_config.pyg
    
    # Determine edge dimension
    edge_dim = pyg_cfg.edge_dim if pyg_cfg.edge_dim else pyg_cfg.edge_in_dim
    
    model = GenericPyGModel(
        node_in_dim=pyg_cfg.node_in_dim,
        edge_in_dim=pyg_cfg.edge_in_dim,
        hid_dim=model_config.message_hidden_dim,
        num_layers=model_config.num_layers,
        gnn_type=pyg_cfg.gnn_type,
        dropout=model_config.dropout,
        pool_type=pyg_cfg.pool_type,
        pool_ratio=pyg_cfg.pool_ratio,
        pool_num_heads=pyg_cfg.pool_num_heads,
        pool_dim_feedforward=pyg_cfg.pool_dim_feedforward,
        num_heads=pyg_cfg.num_heads,
        edge_dim=edge_dim if pyg_cfg.gnn_type == 'transformer' else None,
        ffn_hidden_dim=model_config.ffn_hidden_dim,
        ffn_num_layers=model_config.ffn_num_layers
    )
    
    return model


__all__ = ['GenericPyGModel', 'build_pyg_model']