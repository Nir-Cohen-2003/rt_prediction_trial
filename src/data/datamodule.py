import polars as pl
import lightning as L
from pathlib import Path
from typing import Optional, Callable
import numpy as np
import hashlib
import json
from chemprop.data import MoleculeDatapoint, MoleculeDataset
from chemprop.data.dataloader import build_dataloader
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.featurizers.atom import MultiHotAtomFeaturizer, RIGRAtomFeaturizer
from ..config import DataConfig
from rdkit import Chem
from dataclasses import asdict
import pickle
import torch
import torch_geometric as pyg
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from .lmdb_dataset import LMDBGraphDataset
from .dataset_splitting import split_random, split_scaffold, split_butina

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


class RTDataModule(L.LightningDataModule):
    """
    Lightning DataModule for RT prediction.
    
    Handles data loading, preprocessing, splitting, and batching.
    Supports both Chemprop and custom GNN models with configurable featurizers.
    
    Processed data is saved in data/processed/<hash> where hash is computed
    from the DataConfig to ensure reproducibility and reuse.
    """
    
    def __init__(
        self,
        config: DataConfig,
        model_type: str = "chemprop",
        batch_size: int = 256,
        num_workers: int = 4,
        custom_splitter: Optional[Callable] = None,
        force_rebuild: bool = False
    ):
        """
        Args:
            config: Data configuration
            model_type: "chemprop" or "pyg"
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
        
        # Initialize featurizer
        self.featurizer = self._create_featurizer()
        
        # Will be populated during setup
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Statistics for normalization - NOT optional, must be set during setup
        # Using sentinel values to detect if setup() was called
        self.rt_mean: float = float('nan')
        self.rt_std: float = float('nan')
        
        # Compute hash-based output directory
        self._compute_output_dir()
        
        print(f"[RTDataModule] Initialized with output_dir: {self.output_dir}")
        print(f"[RTDataModule] Using featurizer: {self.config.featurizer_type}")
    
    def _create_featurizer(self):
        """Create the appropriate featurizer based on config."""
        if self.config.featurizer_type == "simple":
            return SimpleMoleculeMolGraphFeaturizer(
                atom_features=self.config.atom_features,
                bond_features=self.config.bond_features,
                atom_descriptors=self.config.atom_descriptors,
                bond_descriptors=self.config.bond_descriptors
            )
        elif self.config.featurizer_type == "v1":
            return SimpleMoleculeMolGraphFeaturizer(atom_featurizer=MultiHotAtomFeaturizer.v1())
        elif self.config.featurizer_type == "v2":
            return SimpleMoleculeMolGraphFeaturizer(atom_featurizer=MultiHotAtomFeaturizer.v2())
        elif self.config.featurizer_type == "organic":
            return SimpleMoleculeMolGraphFeaturizer(atom_featurizer=MultiHotAtomFeaturizer.organic())
        elif self.config.featurizer_type == "rigr":
            return SimpleMoleculeMolGraphFeaturizer(atom_featurizer=RIGRAtomFeaturizer())
        else:
            raise ValueError(f"Unknown featurizer_type: {self.config.featurizer_type}. "
                           f"Must be one of: 'simple', 'v1', 'v2', 'organic', 'rigr'")
    
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
        Creates BOTH Chemprop and PyG datasets for flexibility.
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
        
        # Paths for both dataset types
        chemprop_train_path = self.output_dir / "train_chemprop.pkl"
        chemprop_val_path = self.output_dir / "val_chemprop.pkl"
        chemprop_test_path = self.output_dir / "test_chemprop.pkl"
        lmdb_train_path = self.output_dir / "train_graphs.lmdb"
        lmdb_val_path = self.output_dir / "val_graphs.lmdb"
        lmdb_test_path = self.output_dir / "test_graphs.lmdb"
        
        # Check if all datasets exist
        all_paths_exist = (
            train_path.exists() and 
            stats_path.exists() and
            all(p.exists() for p in [chemprop_train_path, chemprop_val_path, chemprop_test_path]) and
            all(p.exists() for p in [lmdb_train_path, lmdb_val_path, lmdb_test_path])
        )
        
        if all_paths_exist and not self.force_rebuild:
            print(f"[RTDataModule] All processed data already exists in {self.output_dir}, skipping.")
            return
        
        if self.force_rebuild:
            print("[RTDataModule] force_rebuild=True, reprocessing data...")
        
        # Load raw data
        print(f"[RTDataModule] Loading raw data from {self.config.raw_data_path}")
        df = pl.read_csv(self.config.raw_data_path, separator=";")
        
        # Preprocess
        df = preprocess_raw_data(df, self.config)
        
        # Split data based on method
        if self.config.split_method == "random":
            train_df, val_df, test_df = split_random(
                df,
                self.config.test_fraction,
                self.config.val_fraction,
                self.config.random_seed
            )
        elif self.config.split_method == "scaffold":
            train_df, val_df, test_df = split_scaffold(
                df,
                self.config.test_fraction,
                self.config.val_fraction,
                self.config.random_seed,
                self.config.inchi_column
            )
        elif self.config.split_method == "butina":
            train_df, val_df, test_df = split_butina(
                df,
                self.config.test_fraction,
                self.config.val_fraction,
                self.config.random_seed,
                self.config.inchi_column,
                self.config.butina_cutoff,
                self.config.butina_radius,
                self.config.butina_nbits
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
        
        # Compute and save statistics
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
        
        rt_mean = stats["rt_mean"]
        rt_std = stats["rt_std"]
        
        # Create and save BOTH dataset types for flexibility
        print("[RTDataModule] Creating and saving Chemprop datasets...")
        train_chemprop = self._polars_to_chemprop(train_df, rt_mean, rt_std)
        val_chemprop = self._polars_to_chemprop(val_df, rt_mean, rt_std)
        test_chemprop = self._polars_to_chemprop(test_df, rt_mean, rt_std)
        
        with open(chemprop_train_path, "wb") as f:
            pickle.dump(train_chemprop, f)
        with open(chemprop_val_path, "wb") as f:
            pickle.dump(val_chemprop, f)
        with open(chemprop_test_path, "wb") as f:
            pickle.dump(test_chemprop, f)
        print(f"[RTDataModule] Saved Chemprop datasets: {len(train_chemprop)} train, {len(val_chemprop)} val, {len(test_chemprop)} test")
        
        print("[RTDataModule] Creating and saving PyG graphs to LMDB...")
        train_pyg = self._polars_to_pyg(train_df, rt_mean, rt_std)
        val_pyg = self._polars_to_pyg(val_df, rt_mean, rt_std)
        test_pyg = self._polars_to_pyg(test_df, rt_mean, rt_std)
        
        # Save to LMDB
        LMDBGraphDataset.from_graphs(train_pyg, str(lmdb_train_path))
        LMDBGraphDataset.from_graphs(val_pyg, str(lmdb_val_path))
        LMDBGraphDataset.from_graphs(test_pyg, str(lmdb_test_path))
        print(f"[RTDataModule] Saved PyG LMDB datasets: {len(train_pyg)} train, {len(val_pyg)} val, {len(test_pyg)} test")

        print(f"[RTDataModule] prepare_data: END - saved both dataset types to {self.output_dir}")
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for each stage (train/val/test).
        Runs on every GPU/process.
        """
        print(f"[RTDataModule] setup: stage={stage}")
        
        # Load statistics
        stats_path = self.output_dir / "stats.json"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"Statistics file not found at {stats_path}. "
                f"Did prepare_data() run successfully?"
            )
        
        with open(stats_path, "r") as f:
            stats = json.load(f)
        
        # Validate that required statistics exist
        if "rt_mean" not in stats or "rt_std" not in stats:
            raise ValueError(
                f"Statistics file {stats_path} is missing required fields 'rt_mean' or 'rt_std'"
            )
        
        self.rt_mean = float(stats["rt_mean"])
        self.rt_std = float(stats["rt_std"])
        
        # Validate statistics are valid numbers
        if not (0 < self.rt_std < float('inf')):
            raise ValueError(
                f"Invalid RT standard deviation: {self.rt_std}. "
                f"Must be a positive finite number."
            )
        
        print(f"[RTDataModule] Loaded stats: RT mean={self.rt_mean:.2f}, std={self.rt_std:.2f}")
        
        if self.model_type == "chemprop":
            # Load pre-saved Chemprop datasets
            print("[RTDataModule] Loading pre-saved Chemprop datasets...")
            chemprop_train_path = self.output_dir / "train_chemprop.pkl"
            chemprop_val_path = self.output_dir / "val_chemprop.pkl"
            chemprop_test_path = self.output_dir / "test_chemprop.pkl"
            
            if not all(p.exists() for p in [chemprop_train_path, chemprop_val_path, chemprop_test_path]):
                raise FileNotFoundError(
                    f"Chemprop dataset files not found in {self.output_dir}. "
                    f"Did prepare_data() run successfully?"
                )
            
            with open(chemprop_train_path, "rb") as f:
                self.train_dataset = pickle.load(f)
            with open(chemprop_val_path, "rb") as f:
                self.val_dataset = pickle.load(f)
            with open(chemprop_test_path, "rb") as f:
                self.test_dataset = pickle.load(f)
        else:
            # Load LMDB datasets for custom GNN
            print("[RTDataModule] Loading LMDB graph datasets...")
            lmdb_train_path = self.output_dir / "train_graphs.lmdb"
            lmdb_val_path = self.output_dir / "val_graphs.lmdb"
            lmdb_test_path = self.output_dir / "test_graphs.lmdb"
            
            if not all(p.exists() for p in [lmdb_train_path, lmdb_val_path, lmdb_test_path]):
                raise FileNotFoundError(
                    f"LMDB dataset files not found in {self.output_dir}. "
                    f"Did prepare_data() run successfully?"
                )
            
            self.train_dataset = LMDBGraphDataset(str(lmdb_train_path), readonly=True)
            self.val_dataset = LMDBGraphDataset(str(lmdb_val_path), readonly=True)
            self.test_dataset = LMDBGraphDataset(str(lmdb_test_path), readonly=True)
        
        print(f"[RTDataModule] setup: train={len(self.train_dataset)}, "
              f"val={len(self.val_dataset)}, test={len(self.test_dataset)}")
    
    def _polars_to_chemprop(self, df: pl.DataFrame, rt_mean: float, rt_std: float) -> MoleculeDataset:
        """Convert Polars DataFrame to Chemprop MoleculeDataset using configured featurizer."""
        datapoints = []
        skipped = 0
        
        for row in df.iter_rows(named=True):
            inchi = row[self.config.inchi_column]
            rt = row[self.config.rt_column]
            
            # Normalize RT
            rt_norm = (rt - rt_mean) / rt_std
            
            # Convert InChI to RDKit mol with proper sanitization
            mol = Chem.MolFromInchi(inchi, sanitize=False)
            if mol is None:
                skipped += 1
                continue
            
            # Sanitize with charge assignment
            try:
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_PROPERTIES)
                Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
            except Exception as e:
                print(f"[Warning] Failed to sanitize molecule: {e}")
                skipped += 1
                continue
            
            # Create datapoint - featurizer will be applied by Chemprop during batching
            datapoint = MoleculeDatapoint(mol, [rt_norm])
            datapoints.append(datapoint)
        
        if skipped > 0:
            print(f"[Warning] Skipped {skipped} invalid InChI structures")
        
        return MoleculeDataset(datapoints, featurizer=self.featurizer)
    
    def _polars_to_pyg(self, df, rt_mean: float, rt_std: float):
        """Convert Polars DataFrame to PyG dataset using Chemprop featurizer."""
        graphs = []
        skipped = 0
        
        for row in df.iter_rows(named=True):
            inchi = row[self.config.inchi_column]
            rt = row[self.config.rt_column]
            
            # Normalize RT
            rt_norm = (rt - rt_mean) / rt_std
            
            # Convert InChI to RDKit mol with proper sanitization
            mol = Chem.MolFromInchi(inchi, sanitize=False)
            if mol is None:
                skipped += 1
                continue
            
            # Sanitize with charge assignment
            try:
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_PROPERTIES)
                Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
            except Exception as e:
                print(f"[Warning] Failed to sanitize molecule: {e}")
                skipped += 1
                continue
            
            # Get features from Chemprop featurizer
            molgraph = self.featurizer(mol)
            
            # Extract graph structure using correct MolGraph attributes
            # MolGraph has: V (node features), E (edge features), edge_index
            x = torch.tensor(molgraph.V, dtype=torch.float)
            edge_index = torch.from_numpy(molgraph.edge_index)
            
            # Handle edge attributes if present
            if molgraph.E is not None:
                edge_attr = torch.tensor(molgraph.E, dtype=torch.float)
            else:
                edge_attr = None
            
            # Create Data object
            graph = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor([rt_norm], dtype=torch.float)
            )
            graphs.append(graph)
        
        if skipped > 0:
            print(f"[Warning] Skipped {skipped} invalid InChI structures when creating PyG graphs")
        
        return graphs

    def train_dataloader(self):
        """Training dataloader."""
        if self.model_type == "chemprop":
            return build_dataloader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                pin_memory=True,
                persistent_workers=self.num_workers > 0,
            )
        else:
            # Custom GNN with PyG DataLoader
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                pin_memory=True,
                prefetch_factor=4 if self.num_workers > 0 else None,
                persistent_workers=self.num_workers > 0,
            )
    
    def val_dataloader(self):
        """Validation dataloader."""
        if self.model_type == "chemprop":
            return build_dataloader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=True,
                persistent_workers=self.num_workers > 0,
            )
        else:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=True,
                prefetch_factor=4 if self.num_workers > 0 else None,
                persistent_workers=self.num_workers > 0,
            )
    
    def test_dataloader(self):
        """Test dataloader."""
        if self.model_type == "chemprop":
            return build_dataloader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=True,
                persistent_workers=self.num_workers > 0,
            )
        else:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=True,
                prefetch_factor=4 if self.num_workers > 0 else None,
                persistent_workers=self.num_workers > 0,
            )