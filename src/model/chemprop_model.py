import torch
from chemprop.models import MPNN
from chemprop.nn import RegressionFFN, BondMessagePassing
from chemprop.nn.agg import MeanAggregation, SumAggregation, NormAggregation
from chemprop.nn.agg import AttentiveAggregation

from ..config import ModelConfig


def build_chemprop_mpnn(model_config: ModelConfig) -> MPNN:
    """
    Build Chemprop MPNN model from configuration.
    
    This is the single entry point for constructing a Chemprop model.
    It only depends on model configuration, making it reusable and testable.
    
    Args:
        model_config: Model architecture configuration
    
    Returns:
        Configured MPNN model ready for training or inference
    """
    cfg = model_config
    
    # Message passing layer
    message_passing = BondMessagePassing(
        d_h=cfg.message_hidden_dim,
        depth=cfg.num_layers,
        dropout=cfg.dropout,
        activation=cfg.activation,
    )
    
    # Select aggregation function
    if cfg.aggregation == "mean":
        agg = MeanAggregation()
    elif cfg.aggregation == "sum":
        agg = SumAggregation()
    elif cfg.aggregation == "norm":
        agg = NormAggregation()
    elif cfg.aggregation == "attentive":
        agg = AttentiveAggregation(
            input_dim=cfg.message_hidden_dim,
            hidden_dim=cfg.message_hidden_dim,
            dropout=cfg.dropout
        )
    else:
        raise ValueError(f"Unknown aggregation type: {cfg.aggregation}")
    
    # Feed-forward prediction network
    ffn = RegressionFFN(
        input_dim=cfg.message_hidden_dim,
        hidden_dim=cfg.ffn_hidden_dim,
        n_layers=cfg.ffn_num_layers,
        dropout=cfg.dropout,
        activation=cfg.activation,
        n_tasks=1  # Single RT value prediction
    )
    
    # Full MPNN model
    model = MPNN(
        message_passing=message_passing,
        agg=agg,
        predictor=ffn,
        batch_norm=False,
        metrics=None  # We compute metrics manually in training steps
    )
    
    return model