from dataclasses import dataclass, field
from typing import Literal, Optional
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    # Input data
    raw_data_path: Path
    cid_column: str = "cid"
    rt_column: str = "rt"  # in seconds
    inchi_column: str = "inchi"
    
    # Splitting
    split_method: Literal["random", "custom"] = "random"
    test_fraction: float = 0.1
    val_fraction: float = 0.1
    random_seed: int = 42
    
    # Output paths
    output_dir: Path = Path("data/processed")
    train_file: str = "train.parquet"
    val_file: str = "val.parquet"
    test_file: str = "test.parquet"
    
    # Preprocessing flags (you'll implement the logic)
    remove_duplicates: bool = True
    filter_invalid_inchi: bool = True
    min_rt: Optional[float] = None  # filter RT < min_rt
    max_rt: Optional[float] = None  # filter RT > max_rt
    
    def __post_init__(self):
        """Convert string paths to Path objects."""
        self.raw_data_path = Path(self.raw_data_path)
        self.output_dir = Path(self.output_dir)

@dataclass
class deepgcnConfig:
    """Configuration specific to DeepGCN models."""
    
    norm_type: Literal["batch", "layer", "instance"] = "layer"
    beta: float = 1.0  # Initial inverse temperature for GENConv
    learn_beta: bool = True  # Whether beta is learnable
    gen_aggr: Literal["softmax", "power"] = "softmax"  # GENConv aggregation type
    mlp_layers: int = 1  # Number of MLP layers in GENConv
    num_timesteps: int = 2  # For AttentiveFP readout
@dataclass
class PyGModelConfig:
    """Configuration specific to PyTorch Geometric models."""
    
    # Node and edge feature dimensions (usually inferred from data)
    node_in_dim: int = 133  # RDKit default atom features
    edge_in_dim: int = 14   # RDKit default bond features
    
    # Pooling configuration
    pool_type: Literal["mean", "sum", "max", "transformer", "sag", "topk"] = "mean"
    pool_ratio: float = 0.5  # For SAG and TopK pooling
    pool_num_heads: int = 4  # For Transformer pooling
    pool_dim_feedforward: int = 16  # For Transformer pooling
    
    # DeeperGCN specific settings
    deepgcn: deepgcnConfig = field(default_factory=deepgcnConfig)
    
    # Generic GNN settings (for GCN, GAT, GIN, etc.)
    gnn_type: Literal["gcn", "gin","transformer", "deepgcn"] = "gcn"
    num_heads: int = 4  # For  transformer
    edge_dim: Optional[int] = None  # Edge feature dimension for models that support it

@dataclass
class ModelConfig:
    """Configuration for the prediction model (Chemprop or custom GNN)."""
    
    model_type: Literal["chemprop", "gcn", "gat", "gin", "mpnn", "deepgcn", "deep_gcn"] = "chemprop"
    
    # CheMeleon specific settings
    use_chemeleon: bool = False
    chemeleon_checkpoint: Optional[str] = None  # Path or URL to CheMeleon checkpoint
    freeze_chemeleon: bool = False  # Whether to freeze the pretrained encoder
    
    # Common settings for all models
    message_hidden_dim: int = 300
    num_layers: int = 3
    ffn_hidden_dim: int = 300
    ffn_num_layers: int = 2
    dropout: float = 0.0
    activation: Literal["relu", "leakyrelu", "elu"] = "relu"
    
    # Chemprop specific settings
    aggregation: Literal["mean", "sum", "norm", "attentive"] = "mean"
    
    # PyTorch Geometric specific settings
    pyg: PyGModelConfig = field(default_factory=PyGModelConfig)
    
    # For custom GNN (you'll implement)
    custom_gnn_class: Optional[str] = None  # e.g., "SimpleGNNRegressor"
    
    # Input features
    use_additional_features: bool = False
    additional_feature_dim: int = 0
    
    def __post_init__(self):
        """Validate model_type and settings."""
        # attentive is possible only in chemprop
        if self.model_type != "chemprop" and self.aggregation == "attentive":
            raise ValueError("Attentive aggregation is only supported in Chemprop models.")
        
        # Validate CheMeleon settings
        if self.use_chemeleon and self.chemeleon_checkpoint is None:
            raise ValueError("chemeleon_checkpoint must be provided when use_chemeleon=True")
        
        # Ensure pyg config is initialized
        if not isinstance(self.pyg, PyGModelConfig):
            self.pyg = PyGModelConfig(**self.pyg) if isinstance(self.pyg, dict) else PyGModelConfig()


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    
    # Optimization
    learning_rate: float = 1e-4
    batch_size: int = 64
    num_epochs: int = 100
    optimizer: Literal["adam", "adamw", "sgd"] = "adam"
    weight_decay: float = 0.0
    
    # Learning rate scheduling
    use_scheduler: bool = False
    scheduler_type: Literal["plateau", "cosine", "step"] = "plateau"
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    
    # Early stopping
    early_stop_patience: int = 20
    monitor_metric: str = "val/mae"
    monitor_mode: Literal["min", "max"] = "min"
    
    # Checkpointing
    checkpoint_dir: Path = Path("checkpoints")
    save_top_k: int = 3
    
    # Logging
    log_dir: Path = Path("logs")
    log_every_n_steps: int = 10
    
    # Hardware
    accelerator: str = "auto"
    devices: str = "auto"
    precision: Literal["32", "16", "bf16"] = "32"
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False
    
    def __post_init__(self):
        """Convert string paths to Path objects."""
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.log_dir = Path(self.log_dir)


@dataclass
class Config:
    """Top-level configuration combining all components."""
    
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    
    # Experiment metadata
    experiment_name: str = "rt_prediction"
    description: str = ""
    tags: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        from dataclasses import asdict
        return asdict(self)