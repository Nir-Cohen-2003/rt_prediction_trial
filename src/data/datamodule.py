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
from .dataset_splitting import split_random, split_scaffold, split_butina
from .deepgcn_featurizer import get_node_features, get_edge_features

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
        
        # Statistics for normalization - NOT optional, must be set during setup
        # Using sentinel values to detect if setup() was called
        self.rt_mean: float = float('nan')
        self.rt_std: float = float('nan')
        
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
            'rt_column': self.config.rt_column,
            'inchi_column': self.config.inchi_column,
            'split_method': self.config.split_method,
            'test_fraction': self.config.test_fraction,
            'val_fraction': self.config.val_fraction,
            'random_seed': self.config.random_seed,
            'butina_cutoff': self.config.butina_cutoff,
            'butina_radius': self.config.butina_radius,
            'butina_nbits': self.config.butina_nbits,
            'remove_duplicates': self.config.remove_duplicates,
            'filter_invalid_inchi': self.config.filter_invalid_inchi,
            'min_rt': self.config.min_rt,
            'max_rt': self.config.max_rt,
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
                'rt_column': self.config.rt_column,
                'inchi_column': self.config.inchi_column,
                'split_method': self.config.split_method,
                'test_fraction': self.config.test_fraction,
                'val_fraction': self.config.val_fraction,
                'random_seed': self.config.random_seed,
                'butina_cutoff': self.config.butina_cutoff,
                'butina_radius': self.config.butina_radius,
                'butina_nbits': self.config.butina_nbits,
                'remove_duplicates': self.config.remove_duplicates,
                'filter_invalid_inchi': self.config.filter_invalid_inchi,
                'min_rt': self.config.min_rt,
                'max_rt': self.config.max_rt,
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
            train_df, val_df, test_df = self.custom_splitter(df)
        else:
            raise ValueError(f"Unknown split_method: {self.config.split_method}")
        
        # Save splits
        print(f"[RTDataModule] Saving splits to {self.split_dir}...")
        train_df.write_parquet(train_path)
        val_df.write_parquet(val_path)
        test_df.write_parquet(test_path)
        
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
        
        print(f"[RTDataModule] Splits saved: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
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
        rt_mean = stats["rt_mean"]
        rt_std = stats["rt_std"]
        
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
            train_dataset = self._polars_to_chemprop(train_df, rt_mean, rt_std)
            val_dataset = self._polars_to_chemprop(val_df, rt_mean, rt_std)
            test_dataset = self._polars_to_chemprop(test_df, rt_mean, rt_std)
            
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
            train_graphs = self._polars_to_pyg(train_df, rt_mean, rt_std)
            val_graphs = self._polars_to_pyg(val_df, rt_mean, rt_std)
            test_graphs = self._polars_to_pyg(test_df, rt_mean, rt_std)
            
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
            
            # Create datapoint - MoleculeDatapoint expects numpy array for targets
            datapoint = MoleculeDatapoint(mol, np.array([rt_norm]))
            datapoints.append(datapoint)
        
        if skipped > 0:
            print(f"[Warning] Skipped {skipped} invalid InChI structures")
        
        return MoleculeDataset(datapoints, featurizer=self.featurizer)
    
    def _polars_to_pyg(self, df, rt_mean: float, rt_std: float):
        """Convert Polars DataFrame to PyG dataset using configured featurizer."""
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
            
            # Create graph based on featurizer type
            try:
                if self.config.featurizer_type == "rdkit":
                    # Use PyG's built-in from_smiles
                    smiles = Chem.MolToSmiles(mol)
                    graph = from_smiles(smiles)
                    if graph is None:
                        skipped += 1
                        continue
                    # Add y label
                    graph.y = torch.tensor([rt_norm], dtype=torch.float)
                
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
                        y=torch.tensor([rt_norm], dtype=torch.float)
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
                        y=torch.tensor([rt_norm], dtype=torch.float)
                    )
                
                graphs.append(graph)
            
            except Exception as e:
                print(f"[Warning] Failed to create graph: {e}")
                skipped += 1
                continue
        
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