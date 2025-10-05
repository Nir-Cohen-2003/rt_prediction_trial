from dataclasses import dataclass, field, asdict
from typing import Literal, Optional
from pathlib import Path
from chemprop.featurizers.atom import (
                MultiHotAtomFeaturizer,
                RIGRAtomFeaturizer
            )
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
            

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
    
    # Preprocessing flags
    remove_duplicates: bool = True
    filter_invalid_inchi: bool = True
    min_rt: Optional[float] = None
    max_rt: Optional[float] = None
    
    # Molecule featurization - applies to BOTH Chemprop and PyG models
    featurizer_type: Literal["simple", "v1", "v2", "organic", "rigr"] = "rigr"
    
    # SimpleMoleculeFeaturizer options (when featurizer_type="simple")
    atom_features: bool = True
    bond_features: bool = True
    atom_descriptors: Optional[Literal["all", "rdkit_2d", "rdkit_2d_normalized"]] = None
    bond_descriptors: Optional[Literal["all"]] = None
    
    def __post_init__(self):
        """Convert string paths to Path objects."""
        self.raw_data_path = Path(self.raw_data_path)
        self.output_dir = Path(self.output_dir)


@dataclass
class DeepGCNConfig:
    """Configuration specific to DeepGCN models."""
    
    norm_type: Literal["batch", "layer", "instance"] = "layer"
    beta: float = 1.0
    learn_beta: bool = True
    gen_aggr: Literal["softmax", "power"] = "softmax"
    mlp_layers: int = 1
    num_timesteps: int = 2


@dataclass
class PyGModelConfig:
    """Configuration specific to PyTorch Geometric models."""
    
    # Node and edge feature dimensions (will be set from featurizer in __post_init__)
    node_in_dim: int = 133  # Default, will be overridden
    edge_in_dim: int = 14   # Default, will be overridden
    edge_dim: Optional[int] = None  # For TransformerConv, defaults to edge_in_dim if None
    
    # Pooling configuration
    pool_type: Literal["mean", "sum", "max", "transformer", "sag", "topk"] = "mean"
    pool_ratio: float = 0.5
    pool_num_heads: int = 4
    pool_dim_feedforward: int = 16
    
    # DeeperGCN specific settings
    deepgcn: DeepGCNConfig = field(default_factory=DeepGCNConfig)
    
    # Generic GNN settings
    gnn_type: Literal["gcn", "gin", "transformer", "deepgcn"] = "gcn"
    activation: Literal["relu", "silu", "gelu"] = "relu"
    num_heads: int = 4  # For transformer
    use_edge_features: bool = True
    
    def __post_init__(self):
        if self.gnn_type == "deepgcn" and not isinstance(self.deepgcn, DeepGCNConfig):
            self.deepgcn = DeepGCNConfig(**self.deepgcn) if isinstance(self.deepgcn, dict) else self.deepgcn


@dataclass
class ChemPropModelConfig:
    """Configuration specific to Chemprop models."""
    
    aggregation: Literal["mean", "sum", "norm", "attentive"] = "mean"


@dataclass
class ModelConfig:
    """Configuration for the prediction model (Chemprop or PyG)."""

    model_type: Literal["chemprop", "pyg"] = "chemprop"

    # Common settings for ALL models
    message_hidden_dim: int = 64
    num_layers: int = 2
    ffn_hidden_dim: int = 64
    ffn_num_layers: int = 2
    dropout: float = 0.0

    # Model-specific settings (mutually exclusive)
    chemprop: ChemPropModelConfig = field(default_factory=ChemPropModelConfig)
    pyg: PyGModelConfig = field(default_factory=PyGModelConfig)
    
    def __post_init__(self):
        """Validate model_type and initialize appropriate config."""
        if self.model_type == "chemprop":
            if isinstance(self.chemprop, dict):
                self.chemprop = ChemPropModelConfig(**self.chemprop)
        elif self.model_type == "pyg":
            if isinstance(self.pyg, dict):
                self.pyg = PyGModelConfig(**self.pyg)
        else:
            raise ValueError(f"Invalid model_type: {self.model_type}. Must be 'chemprop' or 'pyg'")


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
    
    def __post_init__(self):
        """Set PyG feature dimensions from featurizer configuration."""
        if self.model.model_type == "pyg":
            # Import here to avoid circular dependency
            
            # Create appropriate featurizer to get dimensions
            if self.data.featurizer_type == "simple":
                featurizer = SimpleMoleculeMolGraphFeaturizer(
                    atom_features=self.data.atom_features,
                    bond_features=self.data.bond_features,
                    atom_descriptors=self.data.atom_descriptors,
                    bond_descriptors=self.data.bond_descriptors
                )
            elif self.data.featurizer_type == "v1":
                featurizer = MultiHotAtomFeaturizer.v1()
            elif self.data.featurizer_type == "v2":
                featurizer = MultiHotAtomFeaturizer.v2()
            elif self.data.featurizer_type == "organic":
                featurizer = MultiHotAtomFeaturizer.organic()
            elif self.data.featurizer_type == "rigr":
                featurizer = RIGRAtomFeaturizer()
            else:
                raise ValueError(f"Unknown featurizer_type: {self.data.featurizer_type}")
            
            # Set dimensions in PyG config
            self.model.pyg.node_in_dim = len(featurizer)
            if hasattr(featurizer, 'bond_fdim'):
                self.model.pyg.edge_in_dim = featurizer.bond_fdim
            
            print(f"[Config] Set PyG feature dimensions: node_in_dim={self.model.pyg.node_in_dim}, "
                  f"edge_in_dim={self.model.pyg.edge_in_dim}")
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return asdict(self)