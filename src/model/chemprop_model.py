import torch
from chemprop.models import MPNN
from chemprop.nn import RegressionFFN, BondMessagePassing
from chemprop.nn.agg import MeanAggregation, SumAggregation, NormAggregation
from chemprop.nn.agg import AttentiveAggregation
from pathlib import Path

from ..config import ModelConfig


def build_chemprop_mpnn(model_config: ModelConfig) -> MPNN:
    """
    Build Chemprop MPNN model from configuration.
    
    Args:
        model_config: Model architecture configuration
    
    Returns:
        Configured MPNN model ready for training or inference
    """
    # Extract common and Chemprop-specific config
    cfg = model_config.chemprop
    
    # Check if using CheMeleon pretrained model
    if cfg.use_chemeleon:
        print(f"[build_chemprop_mpnn] Loading CheMeleon pretrained model from {cfg.chemeleon_checkpoint}")
        
        checkpoint_path = Path(cfg.chemeleon_checkpoint)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"CheMeleon checkpoint not found at {checkpoint_path}")
        
        # Load the Lightning checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract the model state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        else:
            raise ValueError("Checkpoint does not contain 'state_dict'")
        
        # Infer architecture from checkpoint or use config override
        try:
            message_hidden_dim = state_dict['message_passing.W_i.weight'].shape[0]
            print(f"[build_chemprop_mpnn] Inferred message_hidden_dim={message_hidden_dim} from checkpoint")
        except KeyError:
            message_hidden_dim = model_config.message_hidden_dim
            print(f"[build_chemprop_mpnn] Using config message_hidden_dim={message_hidden_dim}")
        
        # Infer number of layers
        layer_keys = [k for k in state_dict.keys() if k.startswith('message_passing.') and '.W_i.' in k]
        num_layers = len(layer_keys) if layer_keys else model_config.num_layers
        print(f"[build_chemprop_mpnn] Inferred num_layers={num_layers} from checkpoint")
        
        # Build message passing with inferred/config dimensions
        message_passing = BondMessagePassing(
            d_h=message_hidden_dim,
            depth=num_layers,
            dropout=model_config.dropout,
        )
        
        # Select aggregation function (CheMeleon typically uses mean)
        agg = MeanAggregation()
        
        # Create a temporary FFN (will be replaced)
        temp_ffn = RegressionFFN(
            input_dim=message_hidden_dim,
            hidden_dim=message_hidden_dim,
            n_layers=2,
            dropout=0.0,
            n_tasks=1
        )
        
        # Build the model
        model = MPNN(
            message_passing=message_passing,
            agg=agg,
            predictor=temp_ffn,
            batch_norm=False,
            metrics=None
        )
        
        # Only load encoder weights
        encoder_state_dict = {
            k: v for k, v in state_dict.items() 
            if k.startswith('message_passing.') or k.startswith('agg.')
        }
        
        missing, unexpected = model.load_state_dict(encoder_state_dict, strict=False)
        print(f"[build_chemprop_mpnn] Loaded encoder weights. Missing keys (expected): {len(missing)}")
        
        # Optionally freeze the encoder
        if cfg.freeze_chemeleon:
            print("[build_chemprop_mpnn] Freezing CheMeleon encoder layers")
            for param in model.message_passing.parameters():
                param.requires_grad = False
            for param in model.agg.parameters():
                param.requires_grad = False
        
        # Replace the prediction head
        print("[build_chemprop_mpnn] Creating new prediction head for RT regression")
        model.predictor = RegressionFFN(
            input_dim=message_hidden_dim,
            hidden_dim=model_config.ffn_hidden_dim,
            n_layers=model_config.ffn_num_layers,
            dropout=model_config.dropout,
            n_tasks=1
        )
        
        print(f"[build_chemprop_mpnn] CheMeleon model loaded and adapted for RT prediction")
        return model
    
    # Standard model building (training from scratch)
    print("[build_chemprop_mpnn] Building Chemprop model from scratch")
    
    # Message passing layer
    message_passing = BondMessagePassing(
        d_h=model_config.message_hidden_dim,
        depth=model_config.num_layers,
        dropout=model_config.dropout,
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
            output_size=model_config.message_hidden_dim,
            input_dim=model_config.message_hidden_dim,
            hidden_dim=model_config.message_hidden_dim,
            dropout=model_config.dropout
        )
    else:
        raise ValueError(f"Unknown aggregation type: {cfg.aggregation}")
    
    # Feed-forward prediction network
    ffn = RegressionFFN(
        input_dim=model_config.message_hidden_dim,
        hidden_dim=model_config.ffn_hidden_dim,
        n_layers=model_config.ffn_num_layers,
        dropout=model_config.dropout,
        n_tasks=1
    )
    
    # Full MPNN model
    model = MPNN(
        message_passing=message_passing,
        agg=agg,
        predictor=ffn,
        batch_norm=False,
        metrics=None
    )
    
    return model