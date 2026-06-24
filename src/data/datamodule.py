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
from torch_geometric.utils import from_smiles
from .lmdb_dataset import LMDBGraphDataset
from .dataset_splitting import split_random, split_scaffold, split_butina, split_mces, split_mces_umap
from .deepgcn_featurizer import get_node_features, get_edge_features
from hrms_utils.rdkit import inchi_to_smiles_polars

def preprocess_raw_data(df: pl.DataFrame, config: DataConfig) -> pl.DataFrame:
    """
    Preprocess the raw data.
    
    Args:
        df: Raw dataframe with CID, target columns, and InChI column
        config: Data configuration with preprocessing flags
    
    Returns:
        Cleaned dataframe with added SMILES column
    
    Note: Fill in your custom preprocessing logic here.
    """
    print(f"[preprocess_raw_data] Starting with {len(df)} rows")
    
    # Example preprocessing steps (implement your actual logic):
    
    # 1. Remove duplicates (uses CID column when present)
    if config.remove_duplicates and config.cid_column in df.columns:
        initial_len = len(df)
        df = df.unique(subset=[config.cid_column])
        print(f"[preprocess_raw_data] Removed {initial_len - len(df)} duplicate CIDs")
    
    # 2. Filter invalid InChI (placeholder - add your validation)
    if config.filter_invalid_inchi:
        # TODO: Add InChI validation logic
        # Example: df = df.filter(pl.col(config.inchi_column).str.starts_with("InChI="))
        pass
    
    # 3. Per-target range filters. Null values are preserved so rows missing
    #    one target can still contribute to other targets.
    for col, (min_val, max_val) in config.target_filters.items():
        if col not in df.columns:
            print(f"[preprocess_raw_data] [Warning] target_filters column '{col}' not in dataframe, skipping")
            continue
        if min_val is not None:
            df = df.filter(pl.col(col).is_null() | (pl.col(col) >= min_val))
            print(f"[preprocess_raw_data] Filtered {col} < {min_val} (nulls kept)")
        if max_val is not None:
            df = df.filter(pl.col(col).is_null() | (pl.col(col) <= max_val))
            print(f"[preprocess_raw_data] Filtered {col} > {max_val} (nulls kept)")
    
    # 4. Determine the SMILES source. If a smiles_column is configured and
    #    present in the dataframe, use it directly. Otherwise fall back to
    #    converting the InChI column.
    has_smiles_column = (
        config.smiles_column is not None and config.smiles_column in df.columns
    )
    smiles_source = config.smiles_column if has_smiles_column else config.inchi_column

    # 5. Drop rows with null SMILES source. We deliberately do NOT drop rows
    #    just because a target value is missing; the trainer masks missing targets.
    df = df.drop_nulls(subset=[smiles_source])

    # 6. Materialize a canonical "smiles" column.
    if has_smiles_column:
        print(f"[preprocess_raw_data] Using pre-existing SMILES column '{config.smiles_column}'")
        if config.smiles_column != "smiles":
            df = df.rename({config.smiles_column: "smiles"})
        df = df.filter(
            pl.col("smiles").is_not_null(),
            pl.col("smiles").ne("")
        )
    else:
        print(f"[preprocess_raw_data] Converting InChI to SMILES...")
        df = df.with_columns(
            pl.col(config.inchi_column).map_batches(function=inchi_to_smiles_polars, return_dtype=pl.String, is_elementwise=True).alias("smiles") #type: ignore
        ).filter(
            pl.col("smiles").is_not_null(),
            pl.col("smiles").ne("")
        )

    print(f"[preprocess_raw_data] Successfully prepared {len(df)} molecules with SMILES")
    print(f"[preprocess_raw_data] Finished with {len(df)} rows")
    return df


class RTDataModule(L.LightningDataModule):
    """
    Lightning DataModule for RT prediction.
    
    Handles data loading, preprocessing, splitting, and batching.
    Supports both Chemprop and custom GNN models with configurable featurizers.
    
    Data splitting is cached separately from graph featurization:
    - Split data saved in data/processed/<split_hash>/
    - Featurized graphs saved in data/processed/<split_hash>/graphs_<featurizer>_<model_type>/
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
        
        # Per-target normalization statistics, populated during setup.
        self.target_columns: list[str] = list(config.target_columns)
        self.target_means: dict[str, float] = {}
        self.target_stds: dict[str, float] = {}
        
        # Compute hash-based output directories
        self._compute_output_dirs()
        
        print(f"[RTDataModule] Split data dir: {self.split_dir}")
        print(f"[RTDataModule] Graph data dir: {self.graph_dir}")
        print(f"[RTDataModule] Using featurizer: {self.config.featurizer_type}")
    
    def _create_featurizer(self):
        """Create the appropriate featurizer based on config."""
        if self.config.featurizer_type in ["rdkit", "rdkit_deepgcn"]:
            # These are handled directly in _polars_to_pyg, not via Chemprop
            return None
        elif self.config.featurizer_type == "simple":
            # SimpleMoleculeMolGraphFeaturizer doesn't accept these parameters
            # It uses default atom and bond featurizers
            return SimpleMoleculeMolGraphFeaturizer()
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
                           f"Must be one of: 'simple', 'v1', 'v2', 'organic', 'rigr', 'rdkit', 'rdkit_deepgcn'")
    
    def _compute_output_dirs(self):
        """
        Compute output directories:
        1. Split directory based on data config (excluding featurizer)
        2. Graph directory includes split + featurizer + model_type
        """
        # Create fingerprint for SPLITTING only (exclude featurizer)
        split_config = {
            'raw_data_path': str(self.config.raw_data_path),
            'cid_column': self.config.cid_column,
            'inchi_column': self.config.inchi_column,
            'smiles_column': self.config.smiles_column,
            'csv_separator': self.config.csv_separator,
            'dataset_name': self.config.dataset_name,
            'target_columns': list(self.config.target_columns),
            'target_filters': {k: list(v) for k, v in self.config.target_filters.items()},
            'split_method': self.config.split_method,
            'test_fraction': self.config.test_fraction,
            'val_fraction': self.config.val_fraction,
            'random_seed': self.config.random_seed,
            'butina_cutoff': self.config.butina_cutoff,
            'butina_radius': self.config.butina_radius,
            'butina_nbits': self.config.butina_nbits,
            'mces_initial_threshold': self.config.mces_initial_threshold,
            'mces_min_threshold': self.config.mces_min_threshold,
            'mces_umap_n_components': self.config.mces_umap_n_components,
            'mces_umap_n_neighbors': self.config.mces_umap_n_neighbors,
            'mces_umap_min_dist': self.config.mces_umap_min_dist,
            'mces_umap_hdbscan_min_cluster_size': self.config.mces_umap_hdbscan_min_cluster_size,
            'mces_umap_hdbscan_min_samples': self.config.mces_umap_hdbscan_min_samples,
            'mces_umap_min_ratio': self.config.mces_umap_min_ratio,
            'remove_duplicates': self.config.remove_duplicates,
            'filter_invalid_inchi': self.config.filter_invalid_inchi,
        }

        # Hash for split directory
        split_str = json.dumps(split_config, sort_keys=True, default=str)
        split_hash = hashlib.sha256(split_str.encode()).hexdigest()[:16]
        
        # Base directory for this split
        base_dir = Path("data/processed")
        self.split_dir = base_dir / split_hash
        
        # Graph directory includes featurizer and model type
        graph_suffix = f"graphs_{self.config.featurizer_type}_{self.model_type}"
        self.graph_dir = self.split_dir / graph_suffix
        
        print(f"[RTDataModule] Split hash: {split_hash}")
        print(f"[RTDataModule] Graph suffix: {graph_suffix}")
    
    def prepare_data(self):
        """
        Download and prepare data (runs once on main process).
        Step 1: Load, preprocess, and split data (cached in split_dir)
        Step 2: Create featurized graphs (cached in graph_dir)
        """
        print("[RTDataModule] prepare_data: START")
        
        # ========== STEP 1: Split Data ==========
        self._prepare_splits()
        
        # ========== STEP 2: Featurize Graphs ==========
        self._prepare_graphs()
        
        print(f"[RTDataModule] prepare_data: END")
    
    def _prepare_splits(self):
        """Prepare data splits (independent of featurization)."""
        print("[RTDataModule] Step 1: Preparing splits...")
        
        # Create split directory
        self.split_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the split config
        split_config_path = self.split_dir / "split_config.json"
        if not split_config_path.exists():
            split_config = {
                'raw_data_path': str(self.config.raw_data_path),
                'cid_column': self.config.cid_column,
                'inchi_column': self.config.inchi_column,
                'smiles_column': self.config.smiles_column,
                'csv_separator': self.config.csv_separator,
                'dataset_name': self.config.dataset_name,
                'target_columns': list(self.config.target_columns),
                'target_filters': {k: list(v) for k, v in self.config.target_filters.items()},
                'split_method': self.config.split_method,
                'test_fraction': self.config.test_fraction,
                'val_fraction': self.config.val_fraction,
                'random_seed': self.config.random_seed,
                'butina_cutoff': self.config.butina_cutoff,
                'butina_radius': self.config.butina_radius,
                'butina_nbits': self.config.butina_nbits,
                'mces_initial_threshold': self.config.mces_initial_threshold,
                'mces_min_threshold': self.config.mces_min_threshold,
                'mces_umap_n_components': self.config.mces_umap_n_components,
                'mces_umap_n_neighbors': self.config.mces_umap_n_neighbors,
                'mces_umap_min_dist': self.config.mces_umap_min_dist,
                'mces_umap_hdbscan_min_cluster_size': self.config.mces_umap_hdbscan_min_cluster_size,
                'mces_umap_hdbscan_min_samples': self.config.mces_umap_hdbscan_min_samples,
                'mces_umap_min_ratio': self.config.mces_umap_min_ratio,
                'remove_duplicates': self.config.remove_duplicates,
                'filter_invalid_inchi': self.config.filter_invalid_inchi,
            }
            with open(split_config_path, "w") as f:
                json.dump(split_config, f, indent=2, default=str)
            print(f"[RTDataModule] Saved split config to {split_config_path}")
        
        # Check if splits already exist
        train_path = self.split_dir / self.config.train_file
        val_path = self.split_dir / self.config.val_file
        test_path = self.split_dir / self.config.test_file
        stats_path = self.split_dir / "stats.json"
        
        if all(p.exists() for p in [train_path, val_path, test_path, stats_path]) and not self.force_rebuild:
            print(f"[RTDataModule] Splits already exist in {self.split_dir}, skipping.")
            return
        
        if self.force_rebuild:
            print("[RTDataModule] force_rebuild=True, reprocessing splits...")
        
        # Load raw data, auto-detecting the CSV delimiter if not configured.
        print(f"[RTDataModule] Loading raw data from {self.config.raw_data_path}")
        separator = self.config.csv_separator
        if separator is None:
            with open(self.config.raw_data_path, "r") as f:
                first_line = f.readline()
            if "\t" in first_line:
                separator = "\t"
            elif ";" in first_line:
                separator = ";"
            else:
                separator = ","
            print(f"[RTDataModule] Auto-detected CSV separator: {separator!r}")
        df = pl.read_csv(self.config.raw_data_path, separator=separator)
        
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
                "smiles"  # Use smiles column after conversion
            )
        elif self.config.split_method == "butina":
            train_df, val_df, test_df = split_butina(
                df,
                self.config.test_fraction,
                self.config.val_fraction,
                self.config.random_seed,
                "smiles",  # Use smiles column after conversion
                self.config.butina_cutoff,
                self.config.butina_radius,
                self.config.butina_nbits
            )
        elif self.config.split_method == "mces":
            # Set MCES matrix save path if not provided
            mces_matrix_path = self.config.mces_matrix_save_path
            if mces_matrix_path is None:
                mces_matrix_path = str(self.split_dir / "mces_matrix.npy")
            
            train_df, val_df, test_df, actual_threshold = split_mces(
                df,
                self.config.test_fraction,
                self.config.val_fraction,
                self.config.random_seed,
                "smiles",  # Use smiles column after conversion
                mces_matrix_path
            )
            
            # Save the actual MCES threshold used
            mces_info_path = self.split_dir / "mces_info.json"
            mces_info = {
                "actual_threshold_used": int(actual_threshold),
                "initial_threshold": self.config.mces_initial_threshold,
                "min_threshold": self.config.mces_min_threshold
            }
            with open(mces_info_path, "w") as f:
                json.dump(mces_info, f, indent=2)
            print(f"[RTDataModule] Saved MCES info to {mces_info_path}")
            print(f"[RTDataModule] Actual MCES threshold used: {actual_threshold}")
        
        elif self.config.split_method == "mces_umap":
            # Set MCES bounds matrix save path if not provided
            mces_matrix_path = self.config.mces_matrix_save_path
            if mces_matrix_path is None:
                mces_matrix_path = str(self.split_dir / "mces_bounds_matrix.npy")
            
            train_df, val_df, test_df, bounds_matrix, umap_embedding = split_mces_umap(
                df,
                test_fraction=self.config.test_fraction,
                val_fraction=self.config.val_fraction,
                seed=self.config.random_seed,
                smiles_column="smiles",
                mces_matrix_save_path=mces_matrix_path,
                n_components=self.config.mces_umap_n_components,
                n_neighbors=self.config.mces_umap_n_neighbors,
                min_dist=self.config.mces_umap_min_dist,
                hdbscan_min_cluster_size=self.config.mces_umap_hdbscan_min_cluster_size,
                hdbscan_min_samples=self.config.mces_umap_hdbscan_min_samples,
                min_ratio=self.config.mces_umap_min_ratio,
            )
            
            # Save the bounds matrix and UMAP embedding for reproducibility
            bounds_matrix_path = self.split_dir / "mces_bounds_matrix.npy"
            np.save(bounds_matrix_path, bounds_matrix)
            print(f"[RTDataModule] Saved MCES bounds matrix to {bounds_matrix_path}")
            
            umap_embedding_path = self.split_dir / "mces_umap_embedding.npy"
            np.save(umap_embedding_path, umap_embedding)
            print(f"[RTDataModule] Saved UMAP embedding to {umap_embedding_path}")
            
            # Save mces_umap info json
            mces_umap_info_path = self.split_dir / "mces_umap_info.json"
            mces_umap_info = {
                "random_seed": int(self.config.random_seed),
                "mces_matrix_path": str(mces_matrix_path),
                "umap": {
                    "n_components": int(self.config.mces_umap_n_components),
                    "n_neighbors": (int(self.config.mces_umap_n_neighbors)
                                    if self.config.mces_umap_n_neighbors is not None else None),
                    "min_dist": float(self.config.mces_umap_min_dist),
                },
                "hdbscan": {
                    "min_cluster_size": (int(self.config.mces_umap_hdbscan_min_cluster_size)
                                         if self.config.mces_umap_hdbscan_min_cluster_size is not None else None),
                    "min_samples": int(self.config.mces_umap_hdbscan_min_samples),
                },
                "min_ratio": float(self.config.mces_umap_min_ratio),
            }
            with open(mces_umap_info_path, "w") as f:
                json.dump(mces_umap_info, f, indent=2)
            print(f"[RTDataModule] Saved MCES-UMAP info to {mces_umap_info_path}")
        
        elif self.config.split_method == "custom":
            if self.custom_splitter is None:
                raise ValueError("custom_splitter must be provided for 'custom' split_method")
            train_df, val_df, test_df = self.custom_splitter(df)
        else:
            raise ValueError(f"Unknown split_method: {self.config.split_method}")
    
        # Save splits
        print(f"[RTDataModule] Saving splits to {self.split_dir}...")
        train_df.write_parquet(train_path)
        val_df.write_parquet(val_path)
        test_df.write_parquet(test_path)
        
        # Compute and save per-target statistics on the training set only,
        # using only non-null values per target.
        target_means: dict[str, float] = {}
        target_stds: dict[str, float] = {}
        target_mins: dict[str, float] = {}
        target_maxs: dict[str, float] = {}
        for col in self.config.target_columns:
            if col not in train_df.columns:
                raise ValueError(
                    f"Target column '{col}' not found in training dataframe. "
                    f"Available columns: {train_df.columns}"
                )
            non_null = train_df.filter(pl.col(col).is_not_null())[col]
            if len(non_null) == 0:
                raise ValueError(
                    f"Target column '{col}' has no non-null values in the training set; "
                    f"cannot compute statistics."
                )
            target_means[col] = float(non_null.mean())
            # polars std is sample std (ddof=1) by default; cast to float explicitly
            target_stds[col] = float(non_null.std())
            target_mins[col] = float(non_null.min())
            target_maxs[col] = float(non_null.max())
        
        stats = {
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "target_means": target_means,
            "target_stds": target_stds,
            "target_mins": target_mins,
            "target_maxs": target_maxs,
        }
        
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        
        print(f"[RTDataModule] Splits saved: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        for col in self.config.target_columns:
            print(f"[RTDataModule]   {col}: mean={target_means[col]:.4f}, "
                  f"std={target_stds[col]:.4f}, "
                  f"min={target_mins[col]:.4f}, max={target_maxs[col]:.4f}")
    
    def _prepare_graphs(self):
        """Prepare featurized graphs for the current featurizer and model type."""
        print(f"[RTDataModule] Step 2: Preparing graphs (featurizer={self.config.featurizer_type}, model={self.model_type})...")
        
        # Create graph directory
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        
        # Save graph config
        graph_config_path = self.graph_dir / "graph_config.json"
        if not graph_config_path.exists():
            graph_config = {
                'featurizer_type': self.config.featurizer_type,
                'model_type': self.model_type,
            }
            with open(graph_config_path, "w") as f:
                json.dump(graph_config, f, indent=2)
            print(f"[RTDataModule] Saved graph config to {graph_config_path}")
        
        # Load statistics
        stats_path = self.split_dir / "stats.json"
        with open(stats_path, "r") as f:
            stats = json.load(f)
        target_means: dict[str, float] = {k: float(v) for k, v in stats["target_means"].items()}
        target_stds: dict[str, float] = {k: float(v) for k, v in stats["target_stds"].items()}
        
        # Paths for datasets
        if self.model_type == "chemprop":
            train_path = self.graph_dir / "train_chemprop.pkl"
            val_path = self.graph_dir / "val_chemprop.pkl"
            test_path = self.graph_dir / "test_chemprop.pkl"
            
            if all(p.exists() for p in [train_path, val_path, test_path]) and not self.force_rebuild:
                print(f"[RTDataModule] Chemprop graphs already exist in {self.graph_dir}, skipping.")
                return
            
            # Load split dataframes
            train_df = pl.read_parquet(self.split_dir / self.config.train_file)
            val_df = pl.read_parquet(self.split_dir / self.config.val_file)
            test_df = pl.read_parquet(self.split_dir / self.config.test_file)
            
            # Create Chemprop datasets
            print("[RTDataModule] Creating Chemprop datasets...")
            train_dataset = self._polars_to_chemprop(train_df, target_means, target_stds)
            val_dataset = self._polars_to_chemprop(val_df, target_means, target_stds)
            test_dataset = self._polars_to_chemprop(test_df, target_means, target_stds)
            
            # Save
            with open(train_path, "wb") as f:
                pickle.dump(train_dataset, f)
            with open(val_path, "wb") as f:
                pickle.dump(val_dataset, f)
            with open(test_path, "wb") as f:
                pickle.dump(test_dataset, f)
            
            print(f"[RTDataModule] Saved Chemprop datasets: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
        
        else:  # PyG
            train_path = self.graph_dir / "train_graphs.lmdb"
            val_path = self.graph_dir / "val_graphs.lmdb"
            test_path = self.graph_dir / "test_graphs.lmdb"
            
            if all(p.exists() for p in [train_path, val_path, test_path]) and not self.force_rebuild:
                print(f"[RTDataModule] PyG graphs already exist in {self.graph_dir}, skipping.")
                return
            
            # Load split dataframes
            train_df = pl.read_parquet(self.split_dir / self.config.train_file)
            val_df = pl.read_parquet(self.split_dir / self.config.val_file)
            test_df = pl.read_parquet(self.split_dir / self.config.test_file)
            
            # Create PyG datasets
            print("[RTDataModule] Creating PyG datasets...")
            train_graphs = self._polars_to_pyg(train_df, target_means, target_stds)
            val_graphs = self._polars_to_pyg(val_df, target_means, target_stds)
            test_graphs = self._polars_to_pyg(test_df, target_means, target_stds)
            
            # Save to LMDB
            LMDBGraphDataset.from_graphs(train_graphs, str(train_path))
            LMDBGraphDataset.from_graphs(val_graphs, str(val_path))
            LMDBGraphDataset.from_graphs(test_graphs, str(test_path))
            
            print(f"[RTDataModule] Saved PyG LMDB datasets: {len(train_graphs)} train, {len(val_graphs)} val, {len(test_graphs)} test")
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for each stage (train/val/test).
        Runs on every GPU/process.
        """
        print(f"[RTDataModule] setup: stage={stage}")
        
        # Load statistics
        stats_path = self.split_dir / "stats.json"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"Statistics file not found at {stats_path}. "
                f"Did prepare_data() run successfully?"
            )
        
        with open(stats_path, "r") as f:
            stats = json.load(f)
        
        # Validate that required per-target statistics exist
        if "target_means" not in stats or "target_stds" not in stats:
            raise ValueError(
                f"Statistics file {stats_path} is missing required fields 'target_means' or 'target_stds'"
            )
        
        target_means_raw = stats["target_means"]
        target_stds_raw = stats["target_stds"]
        if not isinstance(target_means_raw, dict) or not isinstance(target_stds_raw, dict):
            raise ValueError(
                f"Statistics file {stats_path} has invalid types for "
                f"'target_means' or 'target_stds' (expected dict)."
            )
        
        self.target_means = {k: float(v) for k, v in target_means_raw.items()}
        self.target_stds = {k: float(v) for k, v in target_stds_raw.items()}
        
        # Validate that every configured target has a mean/std and that each std
        # is a positive finite number.
        for col in self.config.target_columns:
            if col not in self.target_means:
                raise ValueError(
                    f"Statistics file {stats_path} is missing mean for target '{col}'"
                )
            if col not in self.target_stds:
                raise ValueError(
                    f"Statistics file {stats_path} is missing std for target '{col}'"
                )
            std = self.target_stds[col]
            mean = self.target_means[col]
            if not (np.isfinite(std) and std > 0):
                raise ValueError(
                    f"Invalid standard deviation for target '{col}': {std}. "
                    f"Must be a positive finite number."
                )
            if not np.isfinite(mean):
                raise ValueError(
                    f"Invalid mean for target '{col}': {mean}. "
                    f"Must be a finite number."
                )
        
        print(f"[RTDataModule] Loaded per-target stats:")
        for col in self.config.target_columns:
            print(f"[RTDataModule]   {col}: mean={self.target_means[col]:.4f}, "
                  f"std={self.target_stds[col]:.4f}")
        
        if self.model_type == "chemprop":
            # Load pre-saved Chemprop datasets
            print("[RTDataModule] Loading pre-saved Chemprop datasets...")
            train_path = self.graph_dir / "train_chemprop.pkl"
            val_path = self.graph_dir / "val_chemprop.pkl"
            test_path = self.graph_dir / "test_chemprop.pkl"
            
            if not all(p.exists() for p in [train_path, val_path, test_path]):
                raise FileNotFoundError(
                    f"Chemprop dataset files not found in {self.graph_dir}. "
                    f"Did prepare_data() run successfully?"
                )
            
            with open(train_path, "rb") as f:
                self.train_dataset = pickle.load(f)
            with open(val_path, "rb") as f:
                self.val_dataset = pickle.load(f)
            with open(test_path, "rb") as f:
                self.test_dataset = pickle.load(f)
        else:
            # Load LMDB datasets for custom GNN
            print("[RTDataModule] Loading LMDB graph datasets...")
            train_path = self.graph_dir / "train_graphs.lmdb"
            val_path = self.graph_dir / "val_graphs.lmdb"
            test_path = self.graph_dir / "test_graphs.lmdb"
            
            if not all(p.exists() for p in [train_path, val_path, test_path]):
                raise FileNotFoundError(
                    f"LMDB dataset files not found in {self.graph_dir}. "
                    f"Did prepare_data() run successfully?"
                )
            
            self.train_dataset = LMDBGraphDataset(str(train_path), readonly=True)
            self.val_dataset = LMDBGraphDataset(str(val_path), readonly=True)
            self.test_dataset = LMDBGraphDataset(str(test_path), readonly=True)
        
        print(f"[RTDataModule] setup: train={len(self.train_dataset)}, "
              f"val={len(self.val_dataset)}, test={len(self.test_dataset)}")
    
    def _polars_to_chemprop(self, df: pl.DataFrame, target_means: dict, target_stds: dict) -> MoleculeDataset:
        """Convert Polars DataFrame to Chemprop MoleculeDataset using configured featurizer."""
        datapoints = []
        skipped = 0
        target_columns = list(self.config.target_columns)
        num_targets = len(target_columns)
        
        for row in df.iter_rows(named=True):
            smiles = row["smiles"]
            
            # Build normalized target vector of shape (num_targets,). Missing
            # values (None) become np.nan; present values are normalized with
            # their per-target mean/std.
            y = np.empty(num_targets, dtype=np.float64)
            for i, col in enumerate(target_columns):
                val = row.get(col)
                if val is None:
                    y[i] = np.nan
                else:
                    y[i] = (float(val) - target_means[col]) / target_stds[col]
            
            # Convert SMILES to RDKit mol
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                skipped += 1
                continue
            
            # Create datapoint - MoleculeDatapoint expects numpy array for targets
            datapoint = MoleculeDatapoint(mol, y=y)
            datapoints.append(datapoint)
        
        if skipped > 0:
            print(f"[Warning] Skipped {skipped} invalid SMILES structures")
        
        return MoleculeDataset(datapoints, featurizer=self.featurizer)
    
    def _polars_to_pyg(self, df, target_means: dict, target_stds: dict):
        """Convert Polars DataFrame to PyG dataset using configured featurizer."""
        graphs = []
        skipped = 0
        target_columns = list(self.config.target_columns)
        num_targets = len(target_columns)
        
        for row in df.iter_rows(named=True):
            smiles = row["smiles"]
            
            # Build normalized target tensor of shape (num_targets,) and a
            # boolean mask tensor of shape (num_targets,). Missing target
            # values become float('nan') and the corresponding mask entry is
            # False so the trainer can ignore them.
            y_values = []
            y_mask_values = []
            for col in target_columns:
                val = row.get(col)
                if val is None:
                    y_values.append(float('nan'))
                    y_mask_values.append(False)
                else:
                    y_values.append((float(val) - target_means[col]) / target_stds[col])
                    y_mask_values.append(True)
            y_tensor = torch.tensor(y_values, dtype=torch.float)
            y_mask_tensor = torch.tensor(y_mask_values, dtype=torch.bool)
            
            # Convert SMILES to RDKit mol
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                skipped += 1
                continue
            
            # Create graph based on featurizer type
            try:
                if self.config.featurizer_type == "rdkit":
                    # Use PyG's built-in from_smiles
                    graph = from_smiles(smiles)
                    if graph is None:
                        skipped += 1
                        continue
                    # Add y label and mask
                    graph.y = y_tensor
                    graph.y_mask = y_mask_tensor
                
                elif self.config.featurizer_type == "rdkit_deepgcn":
                    # Use DeepGCN featurizer
                    x = torch.from_numpy(get_node_features(mol)).float()
                    
                    # Build edge_index and edge_attr
                    if len(mol.GetBonds()) > 0:
                        edges_list = []
                        edge_features_list = []
                        edge_feat_array = get_edge_features(mol)
                        
                        for idx, bond in enumerate(mol.GetBonds()):
                            i = bond.GetBeginAtomIdx()
                            j = bond.GetEndAtomIdx()
                            # Add edges in both directions
                            edges_list.append((i, j))
                            edge_features_list.append(edge_feat_array[idx])
                            edges_list.append((j, i))
                            edge_features_list.append(edge_feat_array[idx])
                        
                        edge_index = torch.tensor(edges_list, dtype=torch.long).t().contiguous()
                        edge_attr = torch.from_numpy(np.array(edge_features_list, dtype=np.float32))
                    else:
                        edge_index = torch.empty((2, 0), dtype=torch.long)
                        edge_attr = None
                    
                    graph = Data(
                        x=x,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        y=y_tensor,
                        y_mask=y_mask_tensor,
                    )
                
                else:
                    # Use Chemprop featurizer
                    molgraph = self.featurizer(mol)
                    
                    # Extract graph structure using correct MolGraph attributes
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
                        y=y_tensor,
                        y_mask=y_mask_tensor,
                    )
                
                graphs.append(graph)
            
            except Exception as e:
                print(f"[Warning] Failed to create graph: {e}")
                skipped += 1
                continue
        
        if skipped > 0:
            print(f"[Warning] Skipped {skipped} invalid SMILES structures when creating PyG graphs")
        
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