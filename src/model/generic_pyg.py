"""
Generic PyTorch Geometric model builder for various GNN architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from ..config import ModelConfig
from .pyg_components import TransformerPool, SAGPool, TopKPool, get_activation


class GenericPyGModel(nn.Module):
    """Generic PyTorch Geometric model supporting multiple GNN architectures."""
    
    def __init__(self,
                 node_in_dim,
                 edge_in_dim,
                 hid_dim,
                 num_layers,
                 gnn_type='gcn',
                 dropout=0.0,
                 activation='relu',
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
        
        # Get activation function
        self.activation_fn = get_activation(activation)
        
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
                mlp = nn.Sequential(
                    nn.Linear(hid_dim, hid_dim),
                    nn.BatchNorm1d(hid_dim),
                    get_activation(activation),
                    nn.Linear(hid_dim, hid_dim)
                )
                conv = gnn.GINConv(mlp, train_eps=True)
            
            elif self.gnn_type == 'transformer':
                conv = gnn.TransformerConv(
                    in_channels=hid_dim,
                    out_channels=hid_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=hid_dim if edge_dim else None,
                    beta=True
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
            mlp_layers.append(get_activation(activation))
            mlp_layers.append(nn.Dropout(dropout))
        mlp_layers.append(nn.Linear(ffn_hidden_dim, 1))
        
        self.out_mlp = nn.Sequential(*mlp_layers)
    
    def forward(self, data):
        """Forward pass."""
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
            x = self.activation_fn(x)
            
            # Dropout
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection
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
    
    # Validate activation
    activation = pyg_cfg.activation.lower()
    if activation not in ['relu', 'silu', 'gelu']:
        raise ValueError(f"Unsupported activation: '{activation}'. Must be one of: 'relu', 'silu', 'gelu'")
    
    # Determine edge dimension (defaults to edge_in_dim if not specified)
    edge_dim = pyg_cfg.edge_dim if pyg_cfg.edge_dim is not None else pyg_cfg.edge_in_dim
    
    model = GenericPyGModel(
        node_in_dim=pyg_cfg.node_in_dim,
        edge_in_dim=pyg_cfg.edge_in_dim,
        hid_dim=model_config.message_hidden_dim,
        num_layers=model_config.num_layers,
        gnn_type=pyg_cfg.gnn_type,
        dropout=model_config.dropout,
        activation=activation,
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