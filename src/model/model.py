"""
Generic model factory for RT prediction.

This module provides a unified interface for building different types of models
(Chemprop, PyTorch Geometric, etc.) based on configuration.
"""

import torch.nn as nn
from ..config import ModelConfig


def build_model(model_config: ModelConfig) -> nn.Module:
    """
    Generic model factory that builds the appropriate model based on configuration.
    
    This is the single entry point for model construction across the entire codebase.
    It dispatches to specific model builders based on model_type.
    
    Args:
        model_config: Model configuration specifying architecture and type
    
    Returns:
        Configured model ready for training or inference
    
    Raises:
        ValueError: If model_type is not recognized
        NotImplementedError: If model_type is recognized but not yet implemented
    """
    model_type = model_config.model_type.lower()
    
    if model_type == "chemprop":
        from .chemprop_model import build_chemprop_mpnn
        return build_chemprop_mpnn(model_config)
    elif model_type == "pyg":
        if model_config.pyg.gnn_type == "deep_gcn":
            from .deep_gcn_pyg import build_deep_gcn
            return build_deep_gcn(model_config)
        else:
            from .pyg_model import build_pyg_gnn
            return build_pyg_gnn(model_config)
    else:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. "
            f"Supported types: 'chemprop', 'gcn', 'gat', 'gin', 'mpnn', 'deepgcn', 'deep_gcn'"
        )


__all__ = ["build_model"]