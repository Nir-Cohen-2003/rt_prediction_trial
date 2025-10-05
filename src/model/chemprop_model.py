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
    
    This is the single entry point for constructing a Chemprop model.
    It only depends on model configuration, making it reusable and testable.
    Supports both training from scratch and fine-tuning from CheMeleon pretrained models.
    
    Args:
        model_config: Model architecture configuration
    
    Returns:
        Configured MPNN model ready for training or inference
    """
    cfg = model_config
    
    # Check if using CheMeleon pretrained model
    if cfg.use_chemeleon:
        print(f"[build_chemprop_mpnn] Loading CheMeleon pretrained model from {cfg.chemeleon_checkpoint}")
        
        # Load the pretrained CheMeleon model from Lightning checkpoint
        checkpoint_path = Path(cfg.chemeleon_checkpoint)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"CheMeleon checkpoint not found at {checkpoint_path}")
        
        # Load the Lightning checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract the model state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'model.' prefix if present (Lightning adds this)
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        else:
            raise ValueError("Checkpoint does not contain 'state_dict'")
        
        # Infer architecture from checkpoint or use config override
        try:
            # Try to infer dimensions from the state dict
            message_hidden_dim = state_dict['message_passing.W_i.weight'].shape[0]
            print(f"[build_chemprop_mpnn] Inferred message_hidden_dim={message_hidden_dim} from checkpoint")
        except KeyError:
            # Fallback to config value
            message_hidden_dim = cfg.message_hidden_dim
            print(f"[build_chemprop_mpnn] Using config message_hidden_dim={message_hidden_dim}")
        
        # Infer number of layers if not explicitly provided
        if cfg.chemeleon_num_layers is not None:
            num_layers = cfg.chemeleon_num_layers
            print(f"[build_chemprop_mpnn] Using config-specified num_layers={num_layers}")
        else:
            # Count layers in state dict
            layer_keys = [k for k in state_dict.keys() if k.startswith('message_passing.') and '.W_i.' in k]
            num_layers = len(layer_keys) if layer_keys else cfg.num_layers
            print(f"[build_chemprop_mpnn] Inferred num_layers={num_layers} from checkpoint")
        
        # Build message passing with inferred/config dimensions (no activation parameter)
        message_passing = BondMessagePassing(
            d_h=message_hidden_dim,
            depth=num_layers,
            dropout=cfg.dropout,
        )
        
        # Select aggregation function (CheMeleon typically uses mean)
        agg = MeanAggregation()
        
        # Create a temporary FFN (will be replaced, no activation parameter)
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
        
        # Load the state dict (excluding the predictor)
        model_keys = set(model.state_dict().keys())
        pretrained_keys = set(state_dict.keys())
        
        # Only load encoder weights (message_passing and agg)
        encoder_state_dict = {
            k: v for k, v in state_dict.items() 
            if k.startswith('message_passing.') or k.startswith('agg.')
        }
        
        # Load with strict=False to allow missing predictor weights
        missing, unexpected = model.load_state_dict(encoder_state_dict, strict=False)
        print(f"[build_chemprop_mpnn] Loaded encoder weights. Missing keys (expected): {len(missing)}")
        
        # Optionally freeze the encoder
        if cfg.freeze_chemeleon:
            print("[build_chemprop_mpnn] Freezing CheMeleon encoder layers")
            for param in model.message_passing.parameters():
                param.requires_grad = False
            for param in model.agg.parameters():
                param.requires_grad = False
        
        # Replace the prediction head with a new one for RT prediction (no activation parameter)
        print("[build_chemprop_mpnn] Creating new prediction head for RT regression")
        model.predictor = RegressionFFN(
            input_dim=message_hidden_dim,
            hidden_dim=cfg.ffn_hidden_dim,
            n_layers=cfg.ffn_num_layers,
            dropout=cfg.dropout,
            n_tasks=1
        )
        
        print(f"[build_chemprop_mpnn] CheMeleon model loaded and adapted for RT prediction")
        return model
    
    # Standard model building (training from scratch)
    print("[build_chemprop_mpnn] Building Chemprop model from scratch")
    
    # Message passing layer (no activation parameter)
    message_passing = BondMessagePassing(
        d_h=cfg.message_hidden_dim,
        depth=cfg.num_layers,
        dropout=cfg.dropout,
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
    
    # Feed-forward prediction network (no activation parameter)
    ffn = RegressionFFN(
        input_dim=cfg.message_hidden_dim,
        hidden_dim=cfg.ffn_hidden_dim,
        n_layers=cfg.ffn_num_layers,
        dropout=cfg.dropout,
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