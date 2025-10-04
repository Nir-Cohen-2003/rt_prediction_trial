import polars as pl
import lightning as L
from pathlib import Path
from typing import Optional, Callable
import numpy as np
import hashlib
import json
from chemprop.data import MoleculeDatapoint, MoleculeDataset
from chemprop.data.dataloader import build_dataloader
from ..config import DataConfig
from rdkit import Chem
from dataclasses import asdict


def preprocess_raw_data(df: pl.DataFrame, config: DataConfig) -> pl.DataFrame:
    """
    Preprocess the raw RT data.
    
    Args:
        df: Raw dataframe with CID, RT, and InChI columns
        config: Data configuration with preprocessing flags
    
    Returns:
        Cleaned dataframe
    
    Note: Fill in your custom preprocessing logic here.
    """
    print(f"[preprocess_raw_data] Starting with {len(df)} rows")
    
    # Example preprocessing steps (implement your actual logic):
    
    # 1. Remove duplicates
    if config.remove_duplicates:
        initial_len = len(df)
        df = df.unique(subset=[config.cid_column])
        print(f"[preprocess_raw_data] Removed {initial_len - len(df)} duplicate CIDs")
    
    # 2. Filter invalid InChI (placeholder - add your validation)
    if config.filter_invalid_inchi:
        # TODO: Add InChI validation logic
        # Example: df = df.filter(pl.col(config.inchi_column).str.starts_with("InChI="))
        pass
    
    # 3. Filter RT range
    if config.min_rt is not None:
        df = df.filter(pl.col(config.rt_column) >= config.min_rt)
        print(f"[preprocess_raw_data] Filtered RT < {config.min_rt}")
    
    if config.max_rt is not None:
        df = df.filter(pl.col(config.rt_column) <= config.max_rt)
        print(f"[preprocess_raw_data] Filtered RT > {config.max_rt}")
    
    # 4. Remove null values
    df = df.drop_nulls(subset=[config.rt_column, config.inchi_column])
    
    print(f"[preprocess_raw_data] Finished with {len(df)} rows")
    return df


def split_random(
    df: pl.DataFrame,
    test_fraction: float,
    val_fraction: float,
    seed: int
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Random split of data into train/val/test.
    
    Args:
        df: Input dataframe
        test_fraction: Fraction for test set
        val_fraction: Fraction for validation set
        seed: Random seed for reproducibility
    
    Returns:
        train_df, val_df, test_df
    """
    n = len(df)
    
    # Shuffle with seed
    df = df.sample(fraction=1.0, seed=seed, shuffle=True)
    
    # Calculate split indices
    test_size = int(n * test_fraction)
    val_size = int(n * val_fraction)
    train_size = n - test_size - val_size
    
    print(f"[split_random] train={train_size}, val={val_size}, test={test_size}")
    
    # Split
    test_df = df.head(test_size)
    val_df = df.slice(test_size, val_size)
    train_df = df.tail(train_size)
    
    return train_df, val_df, test_df


def split_custom(
    df: pl.DataFrame,
    splitter_fn: Callable,
    **kwargs
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Custom split using a user-provided function.
    
    Args:
        df: Input dataframe
        splitter_fn: Function that takes df and returns (train_df, val_df, test_df)
        **kwargs: Additional arguments for splitter_fn
    
    Returns:
        train_df, val_df, test_df
    """
    print(f"[split_custom] Using custom splitter: {splitter_fn.__name__}")
    return splitter_fn(df, **kwargs)


class RTDataModule(L.LightningDataModule):
    """
    Lightning DataModule for RT prediction.
    
    Handles data loading, preprocessing, splitting, and batching.
    Supports both Chemprop and custom GNN models.
    
    Processed data is saved in data/processed/<hash> where hash is computed
    from the DataConfig to ensure reproducibility and reuse.
    """
    
    def __init__(
        self,
        config: DataConfig,
        model_type: str = "chemprop",
        batch_size: int = 64,
        num_workers: int = 4,
        custom_splitter: Optional[Callable] = None,
        force_rebuild: bool = False
    ):
        """
        Args:
            config: Data configuration
            model_type: "chemprop" or "custom_gnn"
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for data loading
            custom_splitter: Optional custom splitting function
            force_rebuild: If True, force reprocessing even if cached data exists
        """
        super().__init__()
        self.config = config
        self.model_type = model_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.custom_splitter = custom_splitter
        self.force_rebuild = force_rebuild
        
        # Will be populated during setup
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Statistics for normalization
        self.rt_mean: Optional[float] = None
        self.rt_std: Optional[float] = None
        
        # Compute hash-based output directory
        self._compute_output_dir()
        
        print(f"[RTDataModule] Initialized with output_dir: {self.output_dir}")
    
    def _compute_output_dir(self):
        """
        Compute output directory based on hash of DataConfig.
        This ensures that identical configs reuse the same processed data.
        """
        # Create fingerprint from relevant config fields
        # Convert config to dict, excluding output_dir itself to avoid circular dependency
        config_dict = asdict(self.config)
        
        # Remove output_dir from hash computation
        config_dict.pop('output_dir', None)
        config_dict.pop('train_file', None)
        config_dict.pop('val_file', None)
        config_dict.pop('test_file', None)
        
        # Serialize and hash
        config_str = json.dumps(config_dict, sort_keys=True, default=str)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
        
        # Set output directory to data/processed/<hash>
        base_dir = Path("data/processed")
        self.output_dir = base_dir / config_hash
        
        print(f"[RTDataModule] Config hash: {config_hash}")
    
    def prepare_data(self):
        """
        Download and prepare data (runs once on main process).
        Here we load, preprocess, split, and save the data.
        """
        print("[RTDataModule] prepare_data: START")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the config used for this processing
        config_path = self.output_dir / "data_config.json"
        if not config_path.exists():
            with open(config_path, "w") as f:
                json.dump(asdict(self.config), f, indent=2, default=str)
            print(f"[RTDataModule] Saved data config to {config_path}")
        
        # Check if already processed
        train_path = self.output_dir / self.config.train_file
        stats_path = self.output_dir / "stats.json"
        
        if train_path.exists() and stats_path.exists() and not self.force_rebuild:
            print(f"[RTDataModule] Processed data already exists in {self.output_dir}, skipping.")
            return
        
        if self.force_rebuild:
            print("[RTDataModule] force_rebuild=True, reprocessing data...")
        
        # Load raw data
        print(f"[RTDataModule] Loading raw data from {self.config.raw_data_path}")
        df = pl.read_csv(self.config.raw_data_path, separator=";")
        
        # Preprocess
        df = preprocess_raw_data(df, self.config)
        
        # Split data
        if self.config.split_method == "random":
            train_df, val_df, test_df = split_random(
                df,
                self.config.test_fraction,
                self.config.val_fraction,
                self.config.random_seed
            )
        elif self.config.split_method == "custom":
            if self.custom_splitter is None:
                raise ValueError("custom_splitter must be provided for 'custom' split_method")
            train_df, val_df, test_df = split_custom(df, self.custom_splitter)
        else:
            raise ValueError(f"Unknown split_method: {self.config.split_method}")
        
        # Save splits
        print(f"[RTDataModule] Saving splits to {self.output_dir}...")
        train_df.write_parquet(self.output_dir / self.config.train_file)
        val_df.write_parquet(self.output_dir / self.config.val_file)
        test_df.write_parquet(self.output_dir / self.config.test_file)
        
        # Save statistics
        stats = {
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "rt_mean": float(train_df[self.config.rt_column].mean()),
            "rt_std": float(train_df[self.config.rt_column].std()),
            "rt_min": float(train_df[self.config.rt_column].min()),
            "rt_max": float(train_df[self.config.rt_column].max())
        }
        
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        
        print(f"[RTDataModule] prepare_data: END - saved to {self.output_dir}")
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for each stage (train/val/test).
        Runs on every GPU/process.
        """
        print(f"[RTDataModule] setup: stage={stage}")
        
        # Load statistics
        stats_path = self.output_dir / "stats.json"
        with open(stats_path, "r") as f:
            stats = json.load(f)
        self.rt_mean = stats["rt_mean"]
        self.rt_std = stats["rt_std"]
        
        print(f"[RTDataModule] Loaded stats: RT mean={self.rt_mean:.2f}, std={self.rt_std:.2f}")
        
        # Load splits
        train_df = pl.read_parquet(self.output_dir / self.config.train_file)
        val_df = pl.read_parquet(self.output_dir / self.config.val_file)
        test_df = pl.read_parquet(self.output_dir / self.config.test_file)
        
        if self.model_type == "chemprop":
            # Convert to Chemprop format
            self.train_dataset = self._polars_to_chemprop(train_df)
            self.val_dataset = self._polars_to_chemprop(val_df)
            self.test_dataset = self._polars_to_chemprop(test_df)
        else:
            # For custom GNN, store as polars (you'll convert in your model)
            self.train_dataset = train_df
            self.val_dataset = val_df
            self.test_dataset = test_df
        
        print(f"[RTDataModule] setup: train={len(self.train_dataset)}, "
              f"val={len(self.val_dataset)}, test={len(self.test_dataset)}")
    
    def _polars_to_chemprop(self, df: pl.DataFrame) -> MoleculeDataset:
        """Convert Polars DataFrame to Chemprop MoleculeDataset."""
        datapoints = []
        skipped = 0
        
        for row in df.iter_rows(named=True):
            inchi = row[self.config.inchi_column]
            rt = row[self.config.rt_column]
            
            # Normalize RT
            rt_norm = (rt - self.rt_mean) / self.rt_std
            
            # Convert InChI to SMILES with RDKit
            mol = Chem.MolFromInchi(inchi)
            if mol is None:
                skipped += 1
                continue
            
            smiles = Chem.MolToSmiles(mol)
            datapoint = MoleculeDatapoint.from_smi(smiles, rt_norm)
            datapoints.append(datapoint)
        
        if skipped > 0:
            print(f"[Warning] Skipped {skipped} invalid InChI structures")
        
        return MoleculeDataset(datapoints)
    
    def train_dataloader(self):
        """Training dataloader."""
        if self.model_type == "chemprop":
            return build_dataloader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True
            )
        else:
            # Custom GNN - you'll implement this
            raise NotImplementedError("Custom GNN dataloader not implemented yet")
    
    def val_dataloader(self):
        """Validation dataloader."""
        if self.model_type == "chemprop":
            return build_dataloader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False
            )
        else:
            raise NotImplementedError("Custom GNN dataloader not implemented yet")
    
    def test_dataloader(self):
        """Test dataloader."""
        if self.model_type == "chemprop":
            return build_dataloader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False
            )
        else:
            raise NotImplementedError("Custom GNN dataloader not implemented yet")