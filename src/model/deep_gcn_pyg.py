"""
DeeperGCN implementation using PyTorch Geometric.

Adapted from DeeperGCN: All You Need to Train Deeper GCNs
https://arxiv.org/abs/2006.07739
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.utils import softmax
from ..config import ModelConfig
from .pyg_components import TransformerPool, SAGPool, TopKPool


class GENConv(MessagePassing):
    """
    Generalized Message Aggregator with SoftMax and PowerMean aggregation.
    
    Args:
        in_dim: Input feature dimension
        out_dim: Output feature dimension
        aggregator: Type of aggregation ('softmax', 'power')
        beta: Inverse temperature for softmax or power parameter
        learn_beta: Whether beta is learnable
        mlp_layers: Number of MLP layers for message transformation
        norm: Type of normalization ('batch', 'layer', 'instance')
    """
    
    def __init__(self, in_dim, out_dim, aggregator='softmax', beta=1.0, 
                 learn_beta=True, mlp_layers=1, norm='layer'):
        super(GENConv, self).__init__(aggr=None)
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.aggregator = aggregator
        
        if learn_beta:
            self.beta = nn.Parameter(torch.tensor(beta))
        else:
            self.register_buffer('beta', torch.tensor(beta))
        
        # MLP for message transformation
        mlp = []
        for i in range(mlp_layers):
            if i == 0:
                mlp.append(nn.Linear(in_dim, out_dim))
            else:
                mlp.append(nn.Linear(out_dim, out_dim))
            
            if norm == 'batch':
                mlp.append(nn.BatchNorm1d(out_dim))
            elif norm == 'layer':
                mlp.append(nn.LayerNorm(out_dim))
            elif norm == 'instance':
                mlp.append(nn.InstanceNorm1d(out_dim))
            
            mlp.append(nn.ReLU())
        
        self.msg_norm = nn.Sequential(*mlp)
        self.edge_encoder = nn.Linear(in_dim, out_dim)
    
    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, in_dim]
        """
        edge_embedding = self.edge_encoder(edge_attr)
        return self.propagate(edge_index, x=x, edge_attr=edge_embedding)
    
    def message(self, x_j, edge_attr):
        """Construct messages from neighbors."""
        msg = x_j + edge_attr
        return self.msg_norm(msg)
    
    def aggregate(self, inputs, index, dim_size=None):
        """Aggregate messages using softmax or power mean."""
        if self.aggregator == 'softmax':
            # SoftMax aggregation
            out = softmax(inputs * self.beta, index, dim=self.node_dim)
            out = out * inputs
            return torch.zeros_like(inputs).scatter_add_(self.node_dim, 
                                                         index.unsqueeze(-1).expand_as(inputs), 
                                                         out)
        
        elif self.aggregator == 'power':
            # PowerMean aggregation
            min_value, max_value = 1e-7, 1e1
            torch.clamp_(inputs, min_value, max_value)
            output = torch.zeros_like(inputs).scatter_add_(
                self.node_dim,
                index.unsqueeze(-1).expand_as(inputs),
                inputs.pow(self.beta)
            )
            output = output.clamp(min_value, max_value).pow(1.0 / self.beta)
            return output
        
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")


class DeeperGCN(nn.Module):
    """
    DeeperGCN model for molecular property prediction.
    
    Based on "DeeperGCN: All You Need to Train Deeper GCNs"
    https://arxiv.org/abs/2006.07739
    
    Args:
        node_in_dim: Input node feature dimension
        edge_in_dim: Input edge feature dimension
        hid_dim: Hidden dimension
        num_layers: Number of GCN layers
        dropout: Dropout rate
        norm: Type of normalization ('batch', 'layer', 'instance')
        beta: Initial inverse temperature
        learn_beta: Whether beta is learnable
        aggr: Aggregation type ('softmax', 'power')
        mlp_layers: Number of MLP layers in GENConv
        pool_type: Type of pooling ('mean', 'sum', 'max', 'transformer', 'sag', 'topk')
        pool_ratio: Ratio for hierarchical pooling (SAG, TopK)
        pool_num_heads: Number of heads for transformer pooling
        pool_dim_feedforward: Feedforward dimension for transformer pooling
        ffn_hidden_dim: Hidden dimension for output MLP
        ffn_num_layers: Number of layers in output MLP
    """
    
    def __init__(self,
                 node_in_dim,
                 edge_in_dim,
                 hid_dim,
                 num_layers,
                 dropout=0.0,
                 norm='layer',
                 beta=1.0,
                 learn_beta=True,
                 aggr='softmax',
                 mlp_layers=1,
                 pool_type='mean',
                 pool_ratio=0.5,
                 pool_num_heads=4,
                 pool_dim_feedforward=128,
                 ffn_hidden_dim=300,
                 ffn_num_layers=2):
        super(DeeperGCN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.pool_type = pool_type
        
        # Initial encoders
        self.node_encoder = nn.Linear(node_in_dim, hid_dim)
        self.edge_encoder = nn.Linear(edge_in_dim, hid_dim)
        
        # GCN layers with pre-activation residual connections
        self.gcns = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            conv = GENConv(
                in_dim=hid_dim,
                out_dim=hid_dim,
                aggregator=aggr,
                beta=beta,
                learn_beta=learn_beta,
                mlp_layers=mlp_layers,
                norm=norm
            )
            self.gcns.append(conv)
            
            if norm == 'batch':
                self.norms.append(nn.BatchNorm1d(hid_dim))
            elif norm == 'layer':
                self.norms.append(nn.LayerNorm(hid_dim))
            elif norm == 'instance':
                self.norms.append(nn.InstanceNorm1d(hid_dim))
            else:
                raise ValueError(f"Unknown norm type: {norm}")
        
        # Readout/Pooling layer
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
        
        # Output MLP - now uses config values
        mlp_list = []
        for i in range(ffn_num_layers):
            if i == 0:
                mlp_list.append(nn.Linear(hid_dim, ffn_hidden_dim))
            else:
                mlp_list.append(nn.Linear(ffn_hidden_dim, ffn_hidden_dim))
            mlp_list.append(nn.ReLU())
            mlp_list.append(nn.Dropout(dropout))
        mlp_list.append(nn.Linear(ffn_hidden_dim, 1))
        
        self.out_mlp = nn.Sequential(*mlp_list)
    
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
        edge_attr = self.edge_encoder(edge_attr)
        
        # Message passing with residual connections
        for layer in range(self.num_layers):
            # Pre-activation
            h = self.norms[layer](x)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            
            # GCN layer with residual connection
            x = self.gcns[layer](h, edge_index, edge_attr) + x
        
        # Graph-level readout
        if self.pool_type in ['mean', 'sum', 'max']:
            graph_feat = self.readout(x, batch, edge_index, edge_attr)
        else:
            graph_feat = self.readout(x, edge_index, batch, edge_attr)
        
        # Final prediction
        out = self.out_mlp(graph_feat)
        
        return out


def build_deep_gcn(model_config: ModelConfig) -> nn.Module:
    """
    Build DeeperGCN model from configuration.
    
    Args:
        model_config: Model configuration
    
    Returns:
        DeeperGCN model
    """
    pyg_cfg = model_config.pyg
    
    model = DeeperGCN(
        node_in_dim=pyg_cfg.node_in_dim,
        edge_in_dim=pyg_cfg.edge_in_dim,
        hid_dim=model_config.message_hidden_dim,
        num_layers=model_config.num_layers,
        dropout=model_config.dropout,
        norm=pyg_cfg.deepgcn.norm_type,
        beta=pyg_cfg.deepgcn.beta,
        learn_beta=pyg_cfg.deepgcn.learn_beta,
        aggr=pyg_cfg.deepgcn.gen_aggr,
        mlp_layers=pyg_cfg.deepgcn.mlp_layers,
        pool_type=pyg_cfg.pool_type,
        pool_ratio=pyg_cfg.pool_ratio,
        pool_num_heads=pyg_cfg.pool_num_heads,
        pool_dim_feedforward=pyg_cfg.pool_dim_feedforward,
        ffn_hidden_dim=model_config.ffn_hidden_dim,
        ffn_num_layers=model_config.ffn_num_layers
    )
    
    return model


__all__ = ['DeeperGCN', 'GENConv', 'build_deep_gcn']