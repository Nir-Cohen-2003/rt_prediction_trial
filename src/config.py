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
    split_method: Literal["random", "custom", "scaffold", "butina"] = "random"
    test_fraction: float = 0.1
    val_fraction: float = 0.1
    random_seed: int = 42
    
    # Butina clustering parameters
    butina_cutoff: float = 0.35  # Tanimoto distance threshold
    butina_radius: int = 2  # Morgan fingerprint radius
    butina_nbits: int = 2048  # Morgan fingerprint size
    
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
    # Note: "rdkit" and "rdkit_deepgcn" are only valid for PyG models
    featurizer_type: Literal["simple", "v1", "v2", "organic", "rigr", "rdkit", "rdkit_deepgcn"] = "rigr"
    
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
    mlp_layers: int = 1  # Number of MLP layers for GENConv message transformation


@dataclass
class PyGModelConfig:
    """Configuration specific to PyTorch Geometric models."""
    
    # Node and edge feature dimensions (will be set from featurizer in __post_init__)
    node_in_dim: int = 133  # Default, will be overridden
    edge_in_dim: int = 14   # Default, will be overridden
    edge_dim: Optional[int] = None  # For TransformerConv, defaults to edge_in_dim if None
    
    # Pooling configuration
    pool_type: Literal["mean", "sum", "max", "transformer", "sag", "topk", "attentivefp"] = "mean"
    pool_ratio: float = 0.5
    pool_num_heads: int = 4
    pool_dim_feedforward: int = 16
    pool_num_timesteps: int = 2  # For AttentiveFP readout only
    
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
        
        # Validate AttentiveFP readout is only used with DeeperGCN
        if self.pool_type == "attentivefp" and self.gnn_type != "deepgcn":
            raise ValueError(
                f"AttentiveFP readout (pool_type='attentivefp') can only be used with "
                f"DeeperGCN models (gnn_type='deepgcn'), got gnn_type='{self.gnn_type}'"
            )


@dataclass
class ChemPropModelConfig:
    """Configuration specific to Chemprop models."""
    
    aggregation: Literal["mean", "sum", "norm", "attentive"] = "mean"
    use_chemeleon: bool = False
    chemeleon_checkpoint: Optional[Path] = None
    freeze_chemeleon: bool = False

    def __post_init__(self):
        # Normalize checkpoint path if provided
        if self.chemeleon_checkpoint is not None:
            self.chemeleon_checkpoint = Path(self.chemeleon_checkpoint)

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
    
    # Loss function
    loss_fn: Literal["mse", "mae", "huber"] = "mse"
    huber_delta: float = 1.0  # Delta parameter for Huber/Smooth L1 loss
    
    # Learning rate scheduling
    use_scheduler: bool = False
    scheduler_type: Literal["plateau", "cosine", "step", "cosine_warmup"] = "plateau"
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    warmup_epochs: int = 5  # For cosine_warmup scheduler
    
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
    devices: int | str | list[int] = "auto"  # Can be int, str, or list of ints
    precision: Literal["32", "16", "bf16"] = "32"
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False
    
    def __post_init__(self):
        """Convert string paths to Path objects and validate settings."""
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.log_dir = Path(self.log_dir)
        
        # Validate huber_delta
        if self.loss_fn in ["huber", "smooth_l1"] and self.huber_delta <= 0:
            raise ValueError(f"huber_delta must be positive, got {self.huber_delta}")


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
        """Validate featurizer and model type compatibility, set PyG feature dimensions."""
        # Validate that rdkit and rdkit_deepgcn are only used with PyG models
        if self.data.featurizer_type in ["rdkit", "rdkit_deepgcn"] and self.model.model_type != "pyg":
            raise ValueError(
                f"Featurizer '{self.data.featurizer_type}' can only be used with PyG models "
                f"(model_type='pyg'), got model_type='{self.model.model_type}'"
            )
        
        if self.model.model_type == "pyg":
            # Import here to avoid circular dependency
            from torch_geometric.utils import from_smiles
            from .data.deepgcn_featurizer import get_node_dim, get_edge_dim
            
            # Create appropriate featurizer to get dimensions
            if self.data.featurizer_type == "rdkit":
                # Use PyG's built-in from_smiles to get dimensions
                # Test with a simple molecule
                test_graph = from_smiles('C')
                self.model.pyg.node_in_dim = test_graph.x.shape[1]
                if test_graph.edge_attr is not None:
                    self.model.pyg.edge_in_dim = test_graph.edge_attr.shape[1]
                else:
                    self.model.pyg.edge_in_dim = 0
            
            elif self.data.featurizer_type == "rdkit_deepgcn":
                # Use DeepGCN featurizer dimensions
                self.model.pyg.node_in_dim = get_node_dim()
                self.model.pyg.edge_in_dim = get_edge_dim()
            
            elif self.data.featurizer_type == "simple":
                # SimpleMoleculeMolGraphFeaturizer doesn't accept these arguments
                # It uses default atom and bond featurizers internally
                featurizer = SimpleMoleculeMolGraphFeaturizer()
                try:
                    self.model.pyg.node_in_dim = len(featurizer)  # type: ignore[arg-type]
                except TypeError:
                    if hasattr(featurizer, 'atom_fdim'):
                        self.model.pyg.node_in_dim = featurizer.atom_fdim  # type: ignore[attr-defined]
                    else:
                        raise ValueError(f"Could not determine node feature dimension for featurizer: {type(featurizer)}")
                
                if hasattr(featurizer, 'bond_fdim'):
                    self.model.pyg.edge_in_dim = featurizer.bond_fdim  # type: ignore[attr-defined]
            
            elif self.data.featurizer_type == "v1":
                featurizer = SimpleMoleculeMolGraphFeaturizer(atom_featurizer=MultiHotAtomFeaturizer.v1())
                try:
                    self.model.pyg.node_in_dim = len(featurizer)  # type: ignore[arg-type]
                except TypeError:
                    if hasattr(featurizer, 'atom_fdim'):
                        self.model.pyg.node_in_dim = featurizer.atom_fdim  # type: ignore[attr-defined]
                    else:
                        raise ValueError(f"Could not determine node feature dimension for featurizer: {type(featurizer)}")
                
                if hasattr(featurizer, 'bond_fdim'):
                    self.model.pyg.edge_in_dim = featurizer.bond_fdim  # type: ignore[attr-defined]
            
            elif self.data.featurizer_type == "v2":
                featurizer = SimpleMoleculeMolGraphFeaturizer(atom_featurizer=MultiHotAtomFeaturizer.v2())
                try:
                    self.model.pyg.node_in_dim = len(featurizer)  # type: ignore[arg-type]
                except TypeError:
                    if hasattr(featurizer, 'atom_fdim'):
                        self.model.pyg.node_in_dim = featurizer.atom_fdim  # type: ignore[attr-defined]
                    else:
                        raise ValueError(f"Could not determine node feature dimension for featurizer: {type(featurizer)}")
                
                if hasattr(featurizer, 'bond_fdim'):
                    self.model.pyg.edge_in_dim = featurizer.bond_fdim  # type: ignore[attr-defined]
            
            elif self.data.featurizer_type == "organic":
                featurizer = SimpleMoleculeMolGraphFeaturizer(atom_featurizer=MultiHotAtomFeaturizer.organic())
                try:
                    self.model.pyg.node_in_dim = len(featurizer)  # type: ignore[arg-type]
                except TypeError:
                    if hasattr(featurizer, 'atom_fdim'):
                        self.model.pyg.node_in_dim = featurizer.atom_fdim  # type: ignore[attr-defined]
                    else:
                        raise ValueError(f"Could not determine node feature dimension for featurizer: {type(featurizer)}")
                
                if hasattr(featurizer, 'bond_fdim'):
                    self.model.pyg.edge_in_dim = featurizer.bond_fdim  # type: ignore[attr-defined]
            
            elif self.data.featurizer_type == "rigr":
                featurizer = SimpleMoleculeMolGraphFeaturizer(atom_featurizer=RIGRAtomFeaturizer())
                try:
                    self.model.pyg.node_in_dim = len(featurizer)  # type: ignore[arg-type]
                except TypeError:
                    if hasattr(featurizer, 'atom_fdim'):
                        self.model.pyg.node_in_dim = featurizer.atom_fdim  # type: ignore[attr-defined]
                    else:
                        raise ValueError(f"Could not determine node feature dimension for featurizer: {type(featurizer)}")
                
                if hasattr(featurizer, 'bond_fdim'):
                    self.model.pyg.edge_in_dim = featurizer.bond_fdim  # type: ignore[attr-defined]
            
            else:
                raise ValueError(f"Unknown featurizer_type: {self.data.featurizer_type}")
            
            print(f"[Config] Set PyG feature dimensions: node_in_dim={self.model.pyg.node_in_dim}, "
                  f"edge_in_dim={self.model.pyg.edge_in_dim}")
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return asdict(self)