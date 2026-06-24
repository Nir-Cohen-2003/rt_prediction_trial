"""Self-contained marimo notebook for systematic size and architecture comparisons.

Runs the systematic size comparison and systematic architecture comparison
experiments on ``enveda_180.csv`` using only PyTorch Geometric models (no
Chemprop). Targets are normalized per-column using the training set's mean
and standard deviation. Loss is ``mse`` and the monitor metric is
``val/loss``. The notebook is fully self-contained: it does not import from
the repo's ``src`` package and does not depend on ``chemprop``.

Resumable: both comparison functions read their own output CSV and skip
experiments that have already finished. They print pivot tables at the end.
"""

import marimo

__generated_with = "0.23.10"
app = marimo.App()


@app.cell
def _imports():
    import copy
    import gc
    import hashlib
    import json
    import math
    import pickle
    import time
    import traceback
    from collections import defaultdict
    from dataclasses import dataclass, field, asdict
    from itertools import product
    from pathlib import Path
    from typing import Any, Callable, Literal, Mapping, Optional, cast

    import lightning as L
    import lmdb
    import numpy as np
    import polars as pl
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from lightning.pytorch.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
    )
    from lightning.pytorch.loggers import CSVLogger
    from mces_splitting import split_dataset_lower_bound_only, split_dataset_umap
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.ML.Cluster import Butina
    import torch_geometric.nn as gnn
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import (
        MessagePassing,
        global_add_pool,
        global_max_pool,
        global_mean_pool,
    )
    from torch_geometric.utils import from_smiles, softmax

    torch.set_float32_matmul_precision("medium")
    return (
        AllChem,
        Butina,
        CSVLogger,
        Chem,
        DataLoader,
        EarlyStopping,
        F,
        L,
        LearningRateMonitor,
        MessagePassing,
        ModelCheckpoint,
        MurckoScaffold,
        Optional,
        Path,
        asdict,
        cast,
        copy,
        dataclass,
        defaultdict,
        field,
        from_smiles,
        gc,
        global_add_pool,
        global_max_pool,
        global_mean_pool,
        gnn,
        hashlib,
        json,
        lmdb,
        math,
        nn,
        np,
        pickle,
        pl,
        product,
        rdFingerprintGenerator,
        softmax,
        split_dataset_lower_bound_only,
        split_dataset_umap,
        time,
        torch,
        traceback,
    )


@app.cell
def _paths(Path):
    # Locate enveda_180.csv relative to the notebook file, not the current
    # working directory. This makes the notebook work regardless of CWD.
    REPO_ROOT = Path(__file__).resolve().parent
    CSV_PATH = REPO_ROOT / "enveda_180.csv"
    PROCESSED_DIR = REPO_ROOT / "data" / "processed"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    SIZE_RESULTS_CSV = REPO_ROOT / "systematic_size_comparison_notebook_results.csv"
    ARCH_RESULTS_CSV = REPO_ROOT / "systematic_architecture_comparison_notebook_results.csv"

    assert CSV_PATH.exists(), (
        f"Could not find enveda_180.csv at {CSV_PATH}. "
        "Place the notebook in the repo root or adjust CSV_PATH."
    )
    return ARCH_RESULTS_CSV, CSV_PATH, SIZE_RESULTS_CSV


@app.cell
def _config_classes(Optional, asdict, cast, dataclass, field, from_smiles):
    # ---------------------------- DataConfig --------------------------------
    @dataclass
    class DataConfig:
        """Configuration for data loading and preprocessing (PyG only)."""

        raw_data_path: object  # Path or str
        cid_column: str = "cid"
        target_columns: list = field(default_factory=lambda: ["rt"])
        inchi_column: str = "inchi"
        smiles_column: Optional[str] = None
        csv_separator: Optional[str] = None
        dataset_name: str = "default"
        split_method: str = "random"  # one of the five supported
        test_fraction: float = 0.1
        val_fraction: float = 0.1
        random_seed: int = 42
        butina_cutoff: float = 0.35
        butina_radius: int = 2
        butina_nbits: int = 2048
        mces_initial_threshold: int = 10
        mces_min_threshold: int = 1
        mces_matrix_save_path: Optional[str] = None
        mces_umap_n_components: int = 2
        mces_umap_n_neighbors: Optional[int] = None
        mces_umap_min_dist: float = 0.1
        mces_umap_hdbscan_min_cluster_size: Optional[int] = None
        mces_umap_hdbscan_min_samples: int = 1
        mces_umap_min_ratio: float = 0.7
        output_dir: object = None  # Path
        train_file: str = "train.parquet"
        val_file: str = "val.parquet"
        test_file: str = "test.parquet"
        remove_duplicates: bool = True
        filter_invalid_inchi: bool = True
        target_filters: dict = field(default_factory=dict)
        # Only "rdkit" is supported in this notebook.
        featurizer_type: str = "rdkit"

        def __post_init__(self):
            from pathlib import Path as _Path

            self.raw_data_path = _Path(self.raw_data_path)
            if self.output_dir is None:
                self.output_dir = _Path("data/processed")
            self.output_dir = _Path(self.output_dir)
            if self.target_filters:
                self.target_filters = {
                    k: cast(tuple, tuple(v) if isinstance(v, list) else v)
                    for k, v in self.target_filters.items()
                }

    # ----------------------- PyG / DeepGCN configs ---------------------------
    @dataclass
    class DeepGCNConfig:
        norm_type: str = "layer"
        beta: float = 1.0
        learn_beta: bool = True
        gen_aggr: str = "softmax"
        mlp_layers: int = 1

    @dataclass
    class PyGModelConfig:
        node_in_dim: int = 133
        edge_in_dim: int = 14
        edge_dim: Optional[int] = None
        pool_type: str = "mean"
        pool_ratio: float = 0.5
        pool_num_heads: int = 4
        pool_dim_feedforward: int = 16
        pool_num_timesteps: int = 2
        deepgcn: object = field(default_factory=DeepGCNConfig)
        gnn_type: str = "gcn"  # gcn, gat, graphsage, gin, transformer, deepgcn
        activation: str = "relu"
        num_heads: int = 4
        use_edge_features: bool = True

        def __post_init__(self):
            if self.gnn_type == "deepgcn" and not isinstance(self.deepgcn, DeepGCNConfig):
                if isinstance(self.deepgcn, dict):
                    self.deepgcn = DeepGCNConfig(**self.deepgcn)
            if self.pool_type == "attentivefp" and self.gnn_type != "deepgcn":
                raise ValueError(
                    "AttentiveFP readout (pool_type='attentivefp') can only be "
                    "used with DeeperGCN models (gnn_type='deepgcn')."
                )

    # ----------------------- Top-level model + training ----------------------
    @dataclass
    class ModelConfig:
        # PyG only in this notebook.
        model_type: str = "pyg"
        message_hidden_dim: int = 64
        num_layers: int = 2
        ffn_hidden_dim: int = 64
        ffn_num_layers: int = 2
        dropout: float = 0.0
        num_targets: int = 1
        pyg: object = field(default_factory=PyGModelConfig)

        def __post_init__(self):
            if self.model_type != "pyg":
                raise ValueError(
                    f"This notebook only supports model_type='pyg', got '{self.model_type}'"
                )
            if isinstance(self.pyg, dict):
                self.pyg = PyGModelConfig(**self.pyg)

    @dataclass
    class TrainingConfig:
        learning_rate: float = 1e-3
        batch_size: int = 128
        num_epochs: int = 50
        optimizer: str = "adam"
        weight_decay: float = 0.0
        loss_fn: str = "mse"
        huber_delta: float = 1.0
        use_scheduler: bool = False
        scheduler_type: str = "plateau"
        scheduler_patience: int = 10
        scheduler_factor: float = 0.5
        warmup_epochs: int = 5
        early_stop_patience: int = 20
        monitor_metric: str = "val/loss"
        monitor_mode: str = "min"
        checkpoint_dir: object = None
        save_top_k: int = 3
        log_dir: object = None
        log_every_n_steps: int = 10
        accelerator: str = "auto"
        devices: object = "auto"
        precision: str = "32"
        seed: int = 42
        deterministic: bool = False

        def __post_init__(self):
            from pathlib import Path as _Path

            if self.checkpoint_dir is None:
                self.checkpoint_dir = _Path("checkpoints")
            if self.log_dir is None:
                self.log_dir = _Path("logs")
            self.checkpoint_dir = _Path(self.checkpoint_dir)
            self.log_dir = _Path(self.log_dir)

    @dataclass
    class Config:
        data: object  # DataConfig
        model: object  # ModelConfig
        training: object  # TrainingConfig
        experiment_name: str = "rt_prediction"
        description: str = ""
        tags: list = field(default_factory=list)

        def __post_init__(self):
            # Set PyG feature dimensions from rdkit featurizer (from_smiles).
            assert self.model.model_type == "pyg", "Notebook is PyG-only"
            assert self.data.featurizer_type == "rdkit", (
                "Notebook uses featurizer_type='rdkit' only"
            )
            test_graph = from_smiles("C")
            if test_graph.x is None:
                raise ValueError("from_smiles('C') returned a graph with no node features")
            self.model.pyg.node_in_dim = test_graph.x.shape[1]
            if test_graph.edge_attr is not None:
                self.model.pyg.edge_in_dim = test_graph.edge_attr.shape[1]
            else:
                self.model.pyg.edge_in_dim = 0

        def to_dict(self) -> dict:
            return asdict(self)

    return Config, DataConfig, ModelConfig, PyGModelConfig, TrainingConfig


@app.cell
def _splitting(
    AllChem,
    Butina,
    Chem,
    MurckoScaffold,
    defaultdict,
    gc,
    np,
    pl,
    rdFingerprintGenerator,
    split_dataset_lower_bound_only,
    split_dataset_umap,
    time,
    torch,
    traceback,
):
    # --------------------- Splitting strategies (PyG-only) ------------------
    def split_random(df, test_fraction, val_fraction, seed):
        n = len(df)
        df = df.sample(fraction=1.0, seed=seed, shuffle=True)
        test_size = int(n * test_fraction)
        val_size = int(n * val_fraction)
        train_size = n - test_size - val_size
        print(f"[split_random] train={train_size}, val={val_size}, test={test_size}")
        test_df = df.head(test_size)
        val_df = df.slice(test_size, val_size)
        train_df = df.tail(train_size)
        return train_df, val_df, test_df

    def split_scaffold(df, test_fraction, val_fraction, seed, smiles_column="smiles"):
        print("[split_scaffold] Computing Bemis-Murcko scaffolds...")
        scaffolds = []
        valid_indices = []
        for idx, row in enumerate(df.iter_rows(named=True)):
            smiles = row[smiles_column]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"[Warning] Invalid SMILES at index {idx}, skipping")
                continue
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                    mol=mol, includeChirality=False
                )
                scaffolds.append(scaffold)
                valid_indices.append(idx)
            except Exception as e:
                print(f"[Warning] Failed to compute scaffold at index {idx}: {e}")
                continue
        df_valid = df[valid_indices]
        scaffold_to_indices = defaultdict(list)
        for idx, scaffold in enumerate(scaffolds):
            scaffold_to_indices[scaffold].append(idx)
        scaffold_sets = sorted(scaffold_to_indices.values(), key=len, reverse=True)
        print(f"[split_scaffold] Found {len(scaffold_sets)} unique scaffolds")
        if scaffold_sets:
            print(
                f"[split_scaffold] Largest scaffold has {len(scaffold_sets[0])} molecules"
            )
            print(
                f"[split_scaffold] Smallest scaffold has {len(scaffold_sets[-1])} molecules"
            )
        rng = np.random.RandomState(seed)
        rng.shuffle(scaffold_sets)
        n = len(df_valid)
        test_size = int(n * test_fraction)
        val_size = int(n * val_fraction)
        train_indices, val_indices, test_indices = [], [], []
        train_cutoff = n - test_size - val_size
        val_cutoff = n - test_size
        current_size = 0
        for scaffold_set in scaffold_sets:
            if current_size < train_cutoff:
                train_indices.extend(scaffold_set)
            elif current_size < val_cutoff:
                val_indices.extend(scaffold_set)
            else:
                test_indices.extend(scaffold_set)
            current_size += len(scaffold_set)
        train_df = df_valid[train_indices]
        val_df = df_valid[val_indices]
        test_df = df_valid[test_indices]
        print(
            f"[split_scaffold] Final split: train={len(train_df)}, "
            f"val={len(val_df)}, test={len(test_df)}"
        )
        return train_df, val_df, test_df

    def _compute_tanimoto_distance_matrix_torch(
        fingerprints, batch_size=1000, use_gpu=True, device=None
    ):
        overall_start = time.time()
        n = len(fingerprints)
        fp_arrays = []
        for i, fp in enumerate(fingerprints):
            if i % 1000 == 0 and i > 0:
                print(f"Converting {i}/{n}...")
            arr = np.zeros((len(fp),), dtype=np.uint8)
            AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
            fp_arrays.append(arr.astype(bool))
        fp_np = np.stack(fp_arrays)
        if device is None:
            if use_gpu and torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        fps_torch = torch.from_numpy(fp_np).to(device)
        bit_counts = fps_torch.sum(dim=1, dtype=torch.int32)
        print(f"[split_butina] Computing distance matrix on {device}...")
        dist_matrix_np = np.zeros((n, n), dtype=np.float32)
        computation_start = time.time()
        blocks_computed = 0
        total_blocks = sum(
            1 for i in range(0, n, batch_size) for j in range(i, n, batch_size)
        )
        for i in range(0, n, batch_size):
            i_end = min(i + batch_size, n)
            batch_i = fps_torch[i:i_end]
            batch_counts_i = bit_counts[i:i_end]
            for j in range(i, n, batch_size):
                j_end = min(j + batch_size, n)
                batch_j = fps_torch[j:j_end]
                batch_counts_j = bit_counts[j:j_end]
                intersection = (batch_i.unsqueeze(1) & batch_j.unsqueeze(0)).sum(
                    dim=2, dtype=torch.int32
                )
                union = (
                    batch_counts_i.unsqueeze(1)
                    + batch_counts_j.unsqueeze(0)
                    - intersection
                )
                similarity = torch.where(
                    union > 0,
                    intersection.float() / union.float(),
                    torch.zeros_like(union, dtype=torch.float32),
                )
                dist_block = 1.0 - similarity
                dist_block_cpu = dist_block.cpu().numpy()
                dist_matrix_np[i:i_end, j:j_end] = dist_block_cpu
                if i != j:
                    dist_matrix_np[j:j_end, i:i_end] = dist_block_cpu.T
                blocks_computed += 1
                del intersection, union, similarity, dist_block, dist_block_cpu
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if blocks_computed % 10 == 0:
                    progress = int(100 * blocks_computed / total_blocks)
                    print(f"Progress: {progress}% ({blocks_computed}/{total_blocks})")
        print(
            f"[split_butina] Computation completed in {time.time() - computation_start:.2f}s"
        )
        print("[split_butina] Freeing device resources...")
        del fps_torch, bit_counts, fp_np, fp_arrays
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print("[split_butina] Device resources freed")
        return dist_matrix_np

    def split_butina(
        df,
        test_fraction,
        val_fraction,
        seed,
        smiles_column="smiles",
        cutoff=0.35,
        radius=2,
        nbits=2048,
        use_gpu=True,
        batch_size=1000,
    ):
        print(
            f"[split_butina] Computing Morgan fingerprints "
            f"(radius={radius}, nbits={nbits})..."
        )
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
        fingerprints = []
        valid_indices = []
        for idx, row in enumerate(df.iter_rows(named=True)):
            smiles = row[smiles_column]
            if smiles is None or smiles == "":
                print(f"[Warning] Empty or null SMILES at index {idx}, skipping")
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"[Warning] Invalid SMILES at index {idx}, skipping")
                continue
            try:
                fp = morgan_gen.GetFingerprint(mol)
                if fp is None:
                    print(
                        f"[Warning] Failed to generate fingerprint at index {idx}, skipping"
                    )
                    continue
                fingerprints.append(fp)
                valid_indices.append(idx)
            except Exception as e:
                print(f"[Warning] Failed to compute fingerprint at index {idx}: {e}")
                continue
        df_valid = df[valid_indices]
        n = len(df_valid)
        print(f"[split_butina] Successfully computed {n} fingerprints")
        print(f"[split_butina] Computing distance matrix for {n} molecules...")
        distances = []
        if use_gpu:
            print("[split_butina] Using PyTorch GPU-accelerated distance computation")
            try:
                distances = _compute_tanimoto_distance_matrix_torch(
                    fingerprints, batch_size=batch_size, use_gpu=use_gpu
                )
            except Exception as e:
                print(f"[Error] PyTorch GPU computation failed: {e}")
                traceback.print_exc()
                print("[split_butina] Falling back to CPU computation")
                use_gpu = False
        if not use_gpu:
            print("[split_butina] Using CPU distance computation")
            distances = []
            for i in range(n):
                if i % 100 == 0:
                    print(f"[split_butina] Progress: {int(100 * i / n)}%")
                for j in range(i + 1, n):
                    similarity = AllChem.DataStructs.TanimotoSimilarity(
                        fingerprints[i], fingerprints[j]
                    )
                    distances.append(1.0 - similarity)
        print(f"[split_butina] Clustering with cutoff={cutoff}...")
        clusters = Butina.ClusterData(distances, n, cutoff, isDistData=True)
        cluster_list = [list(cluster) for cluster in clusters]
        cluster_list.sort(key=len, reverse=True)
        print(f"[split_butina] Found {len(cluster_list)} clusters")
        if cluster_list:
            print(f"[split_butina] Largest cluster: {len(cluster_list[0])} molecules")
            print(
                f"[split_butina] Smallest cluster: {len(cluster_list[-1])} molecules"
            )
        rng = np.random.RandomState(seed)
        rng.shuffle(cluster_list)
        test_size = int(n * test_fraction)
        val_size = int(n * val_fraction)
        train_indices, val_indices, test_indices = [], [], []
        train_cutoff = n - test_size - val_size
        val_cutoff = n - test_size
        current_size = 0
        for cluster in cluster_list:
            if current_size < train_cutoff:
                train_indices.extend(cluster)
            elif current_size < val_cutoff:
                val_indices.extend(cluster)
            else:
                test_indices.extend(cluster)
            current_size += len(cluster)
        train_df = df_valid[train_indices]
        val_df = df_valid[val_indices]
        test_df = df_valid[test_indices]
        print(
            f"[split_butina] Final split: train={len(train_df)}, "
            f"val={len(val_df)}, test={len(test_df)}"
        )
        return train_df, val_df, test_df

    def split_mces(
        df, test_fraction, val_fraction, seed, smiles_column="smiles",
        mces_matrix_save_path=None,
    ):
        smiles_list = df[smiles_column].to_list()
        train_list, val_list, test_list, threshold = split_dataset_lower_bound_only(
            smiles_list,
            test_fraction=test_fraction,
            validation_fraction=val_fraction,
            initial_distinction_threshold=10,
            min_distinction_threshold=1,
            min_ratio=0.7,
            mces_matrix_save_path=mces_matrix_save_path,
        )
        print(f"[split_mces] Using actual MCES threshold: {threshold}")
        train_df = df.filter(pl.col(smiles_column).is_in(train_list))
        val_df = df.filter(pl.col(smiles_column).is_in(val_list))
        test_df = df.filter(pl.col(smiles_column).is_in(test_list))
        return train_df, val_df, test_df, threshold

    def split_mces_umap(
        df,
        test_fraction,
        val_fraction,
        seed,
        smiles_column="smiles",
        mces_matrix_save_path=None,
        n_components=2,
        n_neighbors=None,
        min_dist=0.1,
        hdbscan_min_cluster_size=None,
        hdbscan_min_samples=1,
        min_ratio=0.7,
    ):
        smiles_list = df[smiles_column].to_list()
        umap_kwargs = {
            "n_components": n_components,
            "min_dist": min_dist,
            "random_state": seed,
        }
        if n_neighbors is not None:
            umap_kwargs["n_neighbors"] = n_neighbors
        hdbscan_kwargs = {"min_samples": hdbscan_min_samples}
        if hdbscan_min_cluster_size is not None:
            hdbscan_kwargs["min_cluster_size"] = hdbscan_min_cluster_size
        (
            train_list,
            val_list,
            test_list,
            bounds_matrix,
            umap_embedding,
        ) = split_dataset_umap(
            smiles_list,
            validation_fraction=val_fraction,
            test_fraction=test_fraction,
            min_ratio=min_ratio,
            mces_matrix_save_path=mces_matrix_save_path,
            hdbscan_kwargs=hdbscan_kwargs,
            **umap_kwargs,
        )
        train_df = df.filter(pl.col(smiles_column).is_in(train_list))
        val_df = df.filter(pl.col(smiles_column).is_in(val_list))
        test_df = df.filter(pl.col(smiles_column).is_in(test_list))
        return train_df, val_df, test_df, bounds_matrix, umap_embedding

    return (
        split_butina,
        split_mces,
        split_mces_umap,
        split_random,
        split_scaffold,
    )


@app.cell
def _lmdb_dataset(lmdb, pickle, torch):
    # ----------------------- LMDBGraphDataset ------------------------------
    class LMDBGraphDataset(torch.utils.data.Dataset):
        """Stores PyG graphs in LMDB; reuses one env per path per process.

        LMDB does not allow multiple environments for the same database file
        within a single process. The original implementation opened a temporary
        environment in ``__init__``, closed it, and lazily reopened it later.
        When Lightning calls ``setup`` again during testing while training
        dataloaders still hold the same LMDB file open, this caused
        "environment already open" errors. Sharing one environment per path
        avoids that conflict.
        """

        _env_cache: dict = {}

        def __init__(self, lmdb_path, readonly=True):
            super().__init__()
            self.lmdb_path = lmdb_path
            self.readonly = readonly
            self.env = self._get_env(lmdb_path, readonly)
            with self.env.begin() as txn:
                length_bytes = txn.get(b"__len__")
                self.length = pickle.loads(length_bytes) if length_bytes else 0

        @classmethod
        def _get_env(cls, lmdb_path, readonly):
            key = (lmdb_path, readonly)
            if key not in cls._env_cache:
                cls._env_cache[key] = lmdb.open(
                    lmdb_path,
                    map_size=2 ** 40,
                    subdir=False,
                    readonly=readonly,
                    lock=not readonly,
                )
            return cls._env_cache[key]

        def _ensure_env(self):
            if self.env is None:
                self.env = self._get_env(self.lmdb_path, self.readonly)

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            self._ensure_env()
            assert self.env is not None, "LMDB environment failed to open"
            with self.env.begin() as txn:
                graph_bytes = txn.get(str(idx).encode())
                if graph_bytes is None:
                    raise IndexError(f"Index {idx} out of range")
                return pickle.loads(graph_bytes)

        def __getstate__(self):
            state = self.__dict__.copy()
            state["env"] = None
            return state

        def __setstate__(self, state):
            self.__dict__.update(state)
            self.env = None

        @staticmethod
        def from_graphs(graphs, lmdb_path, append=False):
            env = lmdb.open(lmdb_path, map_size=2 ** 40, subdir=False, lock=True)
            with env.begin(write=True) as txn:
                length_bytes = txn.get(b"__len__")
                start_idx = (
                    pickle.loads(length_bytes) if (append and length_bytes) else 0
                )
                for i, graph in enumerate(graphs):
                    txn.put(str(start_idx + i).encode(), pickle.dumps(graph))
                txn.put(b"__len__", pickle.dumps(start_idx + len(graphs)))
            env.close()

    return (LMDBGraphDataset,)


@app.cell
def _preprocess(Chem, pl):
    # --------------------- preprocess_raw_data ----------------------------
    def preprocess_raw_data(df, config):
        print(f"[preprocess_raw_data] Starting with {len(df)} rows")
        if config.remove_duplicates and config.cid_column in df.columns:
            initial_len = len(df)
            df = df.unique(subset=[config.cid_column])
            print(
                f"[preprocess_raw_data] Removed {initial_len - len(df)} duplicate CIDs"
            )
        if config.filter_invalid_inchi:
            pass
        for col, (min_val, max_val) in config.target_filters.items():
            if col not in df.columns:
                print(
                    f"[preprocess_raw_data] [Warning] target_filters column "
                    f"'{col}' not in dataframe, skipping"
                )
                continue
            if min_val is not None:
                df = df.filter(pl.col(col).is_null() | (pl.col(col) >= min_val))
                print(f"[preprocess_raw_data] Filtered {col} < {min_val} (nulls kept)")
            if max_val is not None:
                df = df.filter(pl.col(col).is_null() | (pl.col(col) <= max_val))
                print(f"[preprocess_raw_data] Filtered {col} > {max_val} (nulls kept)")

        has_smiles_column = (
            config.smiles_column is not None and config.smiles_column in df.columns
        )
        smiles_source = config.smiles_column if has_smiles_column else config.inchi_column
        assert smiles_source is not None, (
            "Either `smiles_column` or `inchi_column` must be configured in DataConfig"
        )
        df = df.drop_nulls(subset=[smiles_source])
        if has_smiles_column:
            print(
                f"[preprocess_raw_data] Using pre-existing SMILES column "
                f"'{config.smiles_column}'"
            )
            assert config.smiles_column is not None
            smiles_col = config.smiles_column
            if smiles_col != "smiles":
                df = df.rename({smiles_col: "smiles"})
            df = df.filter(
                pl.col("smiles").is_not_null(),
                pl.col("smiles").ne(""),
            )
        else:
            print("[preprocess_raw_data] Converting InChI to SMILES...")
            # Without hrms_utils available, just use RDKit directly.
            smiles_col = config.inchi_column
            df = df.with_columns(
                pl.col(smiles_col)
                .map_elements(
                    lambda s: Chem.MolToSmiles(Chem.MolFromInchi(s))
                    if s is not None and s != ""
                    else None,
                    return_dtype=pl.String,
                )
                .alias("smiles")
            ).filter(
                pl.col("smiles").is_not_null(),
                pl.col("smiles").ne(""),
            )
        # Sanity: drop rows whose SMILES cannot be parsed.
        df = df.filter(
            pl.col("smiles").map_elements(
                lambda s: Chem.MolFromSmiles(s) is not None if s else False,
                return_dtype=pl.Boolean,
            )
        )
        print(
            f"[preprocess_raw_data] Successfully prepared {len(df)} molecules with SMILES"
        )
        return df

    return (preprocess_raw_data,)


@app.cell
def _datamodule(
    DataLoader,
    L,
    LMDBGraphDataset,
    Path,
    cast,
    from_smiles,
    hashlib,
    json,
    np,
    pl,
    preprocess_raw_data,
    split_butina,
    split_mces,
    split_mces_umap,
    split_random,
    split_scaffold,
    torch,
):
    pass  # the unused-binding shim below is removed; no extra aliases needed

    class RTDataModule(L.LightningDataModule):
        """PyG-only Lightning DataModule. Caches splits + LMDB graphs."""

        def __init__(
            self,
            config,
            batch_size=256,
            num_workers=0,
            custom_splitter=None,
            force_rebuild=False,
        ):
            super().__init__()
            self.config = config
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.custom_splitter = custom_splitter
            self.force_rebuild = force_rebuild

            self.train_dataset = None
            self.val_dataset = None
            self.test_dataset = None

            self.target_columns = list(config.target_columns)
            self.target_means = {}
            self.target_stds = {}

            self._compute_output_dirs()
            print(f"[RTDataModule] Split data dir: {self.split_dir}")
            print(f"[RTDataModule] Graph data dir: {self.graph_dir}")
            print(f"[RTDataModule] Using featurizer: {self.config.featurizer_type}")

        def _compute_output_dirs(self):
            split_config = {
                "raw_data_path": str(self.config.raw_data_path),
                "cid_column": self.config.cid_column,
                "inchi_column": self.config.inchi_column,
                "smiles_column": self.config.smiles_column,
                "csv_separator": self.config.csv_separator,
                "dataset_name": self.config.dataset_name,
                "target_columns": list(self.config.target_columns),
                "target_filters": {
                    k: list(v) for k, v in self.config.target_filters.items()
                },
                "split_method": self.config.split_method,
                "test_fraction": self.config.test_fraction,
                "val_fraction": self.config.val_fraction,
                "random_seed": self.config.random_seed,
                "butina_cutoff": self.config.butina_cutoff,
                "butina_radius": self.config.butina_radius,
                "butina_nbits": self.config.butina_nbits,
                "mces_initial_threshold": self.config.mces_initial_threshold,
                "mces_min_threshold": self.config.mces_min_threshold,
                "mces_umap_n_components": self.config.mces_umap_n_components,
                "mces_umap_n_neighbors": self.config.mces_umap_n_neighbors,
                "mces_umap_min_dist": self.config.mces_umap_min_dist,
                "mces_umap_hdbscan_min_cluster_size": self.config.mces_umap_hdbscan_min_cluster_size,
                "mces_umap_hdbscan_min_samples": self.config.mces_umap_hdbscan_min_samples,
                "mces_umap_min_ratio": self.config.mces_umap_min_ratio,
                "remove_duplicates": self.config.remove_duplicates,
                "filter_invalid_inchi": self.config.filter_invalid_inchi,
            }
            split_str = json.dumps(split_config, sort_keys=True, default=str)
            split_hash = hashlib.sha256(split_str.encode()).hexdigest()[:16]
            self.split_dir = Path("data/processed") / split_hash
            graph_suffix = f"graphs_{self.config.featurizer_type}_pyg"
            self.graph_dir = self.split_dir / graph_suffix
            print(f"[RTDataModule] Split hash: {split_hash}")
            print(f"[RTDataModule] Graph suffix: {graph_suffix}")

        def prepare_data(self):
            print("[RTDataModule] prepare_data: START")
            self._prepare_splits()
            self._prepare_graphs()
            print("[RTDataModule] prepare_data: END")

        def _prepare_splits(self):
            print("[RTDataModule] Step 1: Preparing splits...")
            self.split_dir.mkdir(parents=True, exist_ok=True)
            split_config_path = self.split_dir / "split_config.json"
            if not split_config_path.exists():
                split_config = {
                    "raw_data_path": str(self.config.raw_data_path),
                    "cid_column": self.config.cid_column,
                    "inchi_column": self.config.inchi_column,
                    "smiles_column": self.config.smiles_column,
                    "csv_separator": self.config.csv_separator,
                    "dataset_name": self.config.dataset_name,
                    "target_columns": list(self.config.target_columns),
                    "target_filters": {
                        k: list(v) for k, v in self.config.target_filters.items()
                    },
                    "split_method": self.config.split_method,
                    "test_fraction": self.config.test_fraction,
                    "val_fraction": self.config.val_fraction,
                    "random_seed": self.config.random_seed,
                    "butina_cutoff": self.config.butina_cutoff,
                    "butina_radius": self.config.butina_radius,
                    "butina_nbits": self.config.butina_nbits,
                    "mces_initial_threshold": self.config.mces_initial_threshold,
                    "mces_min_threshold": self.config.mces_min_threshold,
                    "mces_umap_n_components": self.config.mces_umap_n_components,
                    "mces_umap_n_neighbors": self.config.mces_umap_n_neighbors,
                    "mces_umap_min_dist": self.config.mces_umap_min_dist,
                    "mces_umap_hdbscan_min_cluster_size": self.config.mces_umap_hdbscan_min_cluster_size,
                    "mces_umap_hdbscan_min_samples": self.config.mces_umap_hdbscan_min_samples,
                    "mces_umap_min_ratio": self.config.mces_umap_min_ratio,
                    "remove_duplicates": self.config.remove_duplicates,
                    "filter_invalid_inchi": self.config.filter_invalid_inchi,
                }
                with open(split_config_path, "w") as f:
                    json.dump(split_config, f, indent=2, default=str)
                print(f"[RTDataModule] Saved split config to {split_config_path}")

            train_path = self.split_dir / self.config.train_file
            val_path = self.split_dir / self.config.val_file
            test_path = self.split_dir / self.config.test_file
            stats_path = self.split_dir / "stats.json"

            if (
                all(p.exists() for p in [train_path, val_path, test_path, stats_path])
                and not self.force_rebuild
            ):
                print(
                    f"[RTDataModule] Splits already exist in {self.split_dir}, skipping."
                )
                return
            if self.force_rebuild:
                print("[RTDataModule] force_rebuild=True, reprocessing splits...")

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
            df = preprocess_raw_data(df, self.config)

            if self.config.split_method == "random":
                train_df, val_df, test_df = split_random(
                    df,
                    self.config.test_fraction,
                    self.config.val_fraction,
                    self.config.random_seed,
                )
            elif self.config.split_method == "scaffold":
                train_df, val_df, test_df = split_scaffold(
                    df,
                    self.config.test_fraction,
                    self.config.val_fraction,
                    self.config.random_seed,
                    "smiles",
                )
            elif self.config.split_method == "butina":
                train_df, val_df, test_df = split_butina(
                    df,
                    self.config.test_fraction,
                    self.config.val_fraction,
                    self.config.random_seed,
                    "smiles",
                    self.config.butina_cutoff,
                    self.config.butina_radius,
                    self.config.butina_nbits,
                )
            elif self.config.split_method == "mces":
                mces_matrix_path = self.config.mces_matrix_save_path
                if mces_matrix_path is None:
                    mces_matrix_path = str(self.split_dir / "mces_matrix.npy")
                train_df, val_df, test_df, actual_threshold = split_mces(
                    df,
                    self.config.test_fraction,
                    self.config.val_fraction,
                    self.config.random_seed,
                    "smiles",
                    mces_matrix_path,
                )
                mces_info_path = self.split_dir / "mces_info.json"
                mces_info = {
                    "actual_threshold_used": int(actual_threshold),
                    "initial_threshold": self.config.mces_initial_threshold,
                    "min_threshold": self.config.mces_min_threshold,
                }
                with open(mces_info_path, "w") as f:
                    json.dump(mces_info, f, indent=2)
                print(f"[RTDataModule] Saved MCES info to {mces_info_path}")
                print(f"[RTDataModule] Actual MCES threshold used: {actual_threshold}")
            elif self.config.split_method == "mces_umap":
                mces_matrix_path = self.config.mces_matrix_save_path
                if mces_matrix_path is None:
                    mces_matrix_path = str(self.split_dir / "mces_bounds_matrix.npy")
                (
                    train_df,
                    val_df,
                    test_df,
                    bounds_matrix,
                    umap_embedding,
                ) = split_mces_umap(
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
                bounds_matrix_path = self.split_dir / "mces_bounds_matrix.npy"
                np.save(bounds_matrix_path, bounds_matrix)
                print(f"[RTDataModule] Saved MCES bounds matrix to {bounds_matrix_path}")
                umap_embedding_path = self.split_dir / "mces_umap_embedding.npy"
                np.save(umap_embedding_path, umap_embedding)
                print(f"[RTDataModule] Saved UMAP embedding to {umap_embedding_path}")
            else:
                raise ValueError(f"Unknown split_method: {self.config.split_method}")

            print(f"[RTDataModule] Saving splits to {self.split_dir}...")
            train_df.write_parquet(train_path)
            val_df.write_parquet(val_path)
            test_df.write_parquet(test_path)

            # Per-target stats from the training set.
            target_means = {}
            target_stds = {}
            target_mins = {}
            target_maxs = {}
            for col in self.config.target_columns:
                if col not in train_df.columns:
                    raise ValueError(
                        f"Target column '{col}' not found in training dataframe. "
                        f"Available columns: {train_df.columns}"
                    )
                non_null = train_df.filter(pl.col(col).is_not_null())[col]
                if len(non_null) == 0:
                    raise ValueError(
                        f"Target column '{col}' has no non-null values in the training set"
                    )
                mean_val = non_null.mean()
                std_val = non_null.std()
                min_val = non_null.min()
                max_val = non_null.max()
                assert mean_val is not None
                assert std_val is not None
                assert min_val is not None
                assert max_val is not None
                target_means[col] = cast(float, mean_val)
                target_stds[col] = cast(float, std_val)
                target_mins[col] = cast(float, min_val)
                target_maxs[col] = cast(float, max_val)
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
            print(
                f"[RTDataModule] Splits saved: train={len(train_df)}, "
                f"val={len(val_df)}, test={len(test_df)}"
            )
            for col in self.config.target_columns:
                print(
                    f"[RTDataModule]   {col}: mean={target_means[col]:.4f}, "
                    f"std={target_stds[col]:.4f}, "
                    f"min={target_mins[col]:.4f}, max={target_maxs[col]:.4f}"
                )

        def _prepare_graphs(self):
            print(
                f"[RTDataModule] Step 2: Preparing graphs "
                f"(featurizer={self.config.featurizer_type}, model=pyg)..."
            )
            self.graph_dir.mkdir(parents=True, exist_ok=True)
            graph_config_path = self.graph_dir / "graph_config.json"
            if not graph_config_path.exists():
                graph_config = {
                    "featurizer_type": self.config.featurizer_type,
                    "model_type": "pyg",
                }
                with open(graph_config_path, "w") as f:
                    json.dump(graph_config, f, indent=2)
                print(f"[RTDataModule] Saved graph config to {graph_config_path}")

            stats_path = self.split_dir / "stats.json"
            with open(stats_path, "r") as f:
                stats = json.load(f)
            target_means = {k: float(v) for k, v in stats["target_means"].items()}
            target_stds = {k: float(v) for k, v in stats["target_stds"].items()}

            train_path = self.graph_dir / "train_graphs.lmdb"
            val_path = self.graph_dir / "val_graphs.lmdb"
            test_path = self.graph_dir / "test_graphs.lmdb"
            if (
                all(p.exists() for p in [train_path, val_path, test_path])
                and not self.force_rebuild
            ):
                print(
                    f"[RTDataModule] PyG graphs already exist in {self.graph_dir}, skipping."
                )
                return

            train_df = pl.read_parquet(self.split_dir / self.config.train_file)
            val_df = pl.read_parquet(self.split_dir / self.config.val_file)
            test_df = pl.read_parquet(self.split_dir / self.config.test_file)

            print("[RTDataModule] Creating PyG datasets...")
            train_graphs = self._polars_to_pyg(train_df, target_means, target_stds)
            val_graphs = self._polars_to_pyg(val_df, target_means, target_stds)
            test_graphs = self._polars_to_pyg(test_df, target_means, target_stds)

            LMDBGraphDataset.from_graphs(train_graphs, str(train_path))
            LMDBGraphDataset.from_graphs(val_graphs, str(val_path))
            LMDBGraphDataset.from_graphs(test_graphs, str(test_path))
            print(
                f"[RTDataModule] Saved PyG LMDB datasets: "
                f"{len(train_graphs)} train, {len(val_graphs)} val, "
                f"{len(test_graphs)} test"
            )

        def setup(self, stage=None):
            print(f"[RTDataModule] setup: stage={stage}")
            stats_path = self.split_dir / "stats.json"
            if not stats_path.exists():
                raise FileNotFoundError(
                    f"Statistics file not found at {stats_path}. "
                    f"Did prepare_data() run successfully?"
                )
            with open(stats_path, "r") as f:
                stats = json.load(f)
            if "target_means" not in stats or "target_stds" not in stats:
                raise ValueError(
                    f"Statistics file {stats_path} is missing required fields"
                )
            self.target_means = {k: float(v) for k, v in stats["target_means"].items()}
            self.target_stds = {k: float(v) for k, v in stats["target_stds"].items()}
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
                        f"Invalid standard deviation for target '{col}': {std}"
                    )
                if not np.isfinite(mean):
                    raise ValueError(f"Invalid mean for target '{col}': {mean}")
            print("[RTDataModule] Loaded per-target stats:")
            for col in self.config.target_columns:
                print(
                    f"[RTDataModule]   {col}: mean={self.target_means[col]:.4f}, "
                    f"std={self.target_stds[col]:.4f}"
                )

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
            print(
                f"[RTDataModule] setup: train={len(self.train_dataset)}, "
                f"val={len(self.val_dataset)}, test={len(self.test_dataset)}"
            )

        def _polars_to_pyg(self, df, target_means, target_stds):
            assert self.config.featurizer_type == "rdkit", (
                "Notebook only supports featurizer_type='rdkit'"
            )
            graphs = []
            skipped = 0
            target_columns = list(self.config.target_columns)
            num_targets = len(target_columns)
            for row in df.iter_rows(named=True):
                smiles = row["smiles"]
                y_values = []
                y_mask_values = []
                for col in target_columns:
                    val = row.get(col)
                    if val is None:
                        y_values.append(float("nan"))
                        y_mask_values.append(False)
                    else:
                        y_values.append(
                            (float(val) - target_means[col]) / target_stds[col]
                        )
                        y_mask_values.append(True)
                y_tensor = torch.tensor(y_values, dtype=torch.float)
                y_mask_tensor = torch.tensor(y_mask_values, dtype=torch.bool)
                try:
                    graph = from_smiles(smiles)
                    if graph is None:
                        skipped += 1
                        continue
                    graph.y = y_tensor
                    graph.y_mask = y_mask_tensor
                    graphs.append(graph)
                except Exception as e:
                    print(f"[Warning] Failed to create graph: {e}")
                    skipped += 1
                    continue
            if skipped > 0:
                print(
                    f"[Warning] Skipped {skipped} invalid SMILES structures "
                    f"when creating PyG graphs"
                )
            return graphs

        def train_dataloader(self):
            assert self.train_dataset is not None
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
            assert self.val_dataset is not None
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
            assert self.test_dataset is not None
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=True,
                prefetch_factor=4 if self.num_workers > 0 else None,
                persistent_workers=self.num_workers > 0,
            )

    return (RTDataModule,)


@app.cell
def _pyg_components(global_add_pool, global_mean_pool, nn, softmax, torch):
    # ----------------------- PyG pooling components -------------------------

    class TransformerPool(nn.Module):
        def __init__(self, in_channels, num_heads=4, dim_feedforward=128, dropout_rate=0.1):
            super().__init__()
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=in_channels,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout_rate,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        def forward(self, x, edge_index, batch, edge_attr=None):
            attn_mask = batch.unsqueeze(0) != batch.unsqueeze(1)
            x = x.unsqueeze(0)
            x = self.encoder(x, mask=attn_mask)
            x = x.squeeze(0)
            is_first = torch.cat(
                [torch.tensor([True], device=x.device), batch[1:] != batch[:-1]]
            )
            return x[is_first]

    class SAGPool(nn.Module):
        def __init__(self, in_channels, ratio=0.5):
            super().__init__()
            from torch_geometric.nn import SAGPooling, global_mean_pool as _gmp

            self.pool = SAGPooling(in_channels, ratio)
            self._gmp = _gmp

        def forward(self, x, edge_index, batch, edge_attr=None):
            x, edge_index, edge_attr, batch, _, _ = self.pool(
                x, edge_index, edge_attr, batch
            )
            return self._gmp(x, batch)

    class TopKPool(nn.Module):
        def __init__(self, in_channels, ratio=0.5):
            super().__init__()
            from torch_geometric.nn import TopKPooling, global_mean_pool as _gmp

            self.pool = TopKPooling(in_channels, ratio)
            self._gmp = _gmp

        def forward(self, x, edge_index, batch, edge_attr=None):
            x, edge_index, edge_attr, batch, _, _ = self.pool(
                x, edge_index, edge_attr, batch
            )
            return self._gmp(x, batch)

    def get_activation(name):
        if name == "relu":
            return nn.ReLU()
        elif name == "silu":
            return nn.SiLU()
        elif name == "gelu":
            return nn.GELU()
        else:
            raise ValueError(
                f"Unknown activation function: {name}, supported: 'relu', 'silu', 'gelu'"
            )

    class AttentiveFPReadout(nn.Module):
        """AttentiveFP Graph Readout with GRU-based attention."""

        def __init__(self, feat_size, num_timesteps=2, dropout=0.0):
            super().__init__()
            self.feat_size = feat_size
            self.num_timesteps = num_timesteps
            self.gru = nn.GRUCell(feat_size, feat_size)
            self.attend = nn.Linear(feat_size, feat_size, bias=False)
            self.dropout = nn.Dropout(dropout)

        def forward(self, node_feats, batch):
            batch_size = int(batch.max().item()) + 1
            graph_feats = global_mean_pool(node_feats, batch)
            for _ in range(self.num_timesteps):
                graph_feats_expanded = graph_feats[batch]
                attended_node = self.attend(node_feats)
                attention_scores = (attended_node * graph_feats_expanded).sum(dim=-1)
                attention_weights = softmax(attention_scores, batch, dim=0)
                weighted_feats = node_feats * attention_weights.unsqueeze(-1)
                context = global_add_pool(weighted_feats, batch)
                context = self.dropout(context)
                graph_feats = self.gru(context, graph_feats)
            return graph_feats

    return (
        AttentiveFPReadout,
        SAGPool,
        TopKPool,
        TransformerPool,
        get_activation,
    )


@app.cell
def _generic_pyg(
    F,
    SAGPool,
    TopKPool,
    TransformerPool,
    get_activation,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
    gnn,
    nn,
):
    # ----------------------- Generic PyG model ------------------------------

    class GenericPyGModel(nn.Module):
        def __init__(
            self,
            node_in_dim,
            edge_in_dim,
            hid_dim,
            num_layers,
            gnn_type="gcn",
            dropout=0.0,
            activation="relu",
            pool_type="mean",
            pool_ratio=0.5,
            pool_num_heads=4,
            pool_dim_feedforward=128,
            num_heads=4,
            edge_dim=None,
            ffn_hidden_dim=300,
            ffn_num_layers=2,
            num_targets=1,
            use_edge_features=True,
        ):
            super().__init__()
            self.num_layers = num_layers
            self.dropout = dropout
            self.gnn_type = gnn_type.lower()
            self.pool_type = pool_type
            self.use_edge_features = use_edge_features
            self.activation_fn = get_activation(activation)

            self.node_encoder = nn.Linear(node_in_dim, hid_dim)
            self.edge_encoder = nn.Linear(edge_in_dim, hid_dim) if edge_dim else None

            self.convs = nn.ModuleList()
            self.batch_norms = nn.ModuleList()
            for _ in range(num_layers):
                if self.gnn_type == "gcn":
                    conv = gnn.GCNConv(hid_dim, hid_dim)
                elif self.gnn_type == "gin":
                    mlp = nn.Sequential(
                        nn.Linear(hid_dim, hid_dim),
                        nn.BatchNorm1d(hid_dim),
                        get_activation(activation),
                        nn.Linear(hid_dim, hid_dim),
                    )
                    conv = gnn.GINConv(mlp, train_eps=True)
                elif self.gnn_type == "transformer":
                    conv = gnn.TransformerConv(
                        in_channels=hid_dim,
                        out_channels=hid_dim // num_heads,
                        heads=num_heads,
                        dropout=dropout,
                        edge_dim=hid_dim if edge_dim else None,
                        beta=True,
                    )
                elif self.gnn_type == "gat":
                    conv = gnn.GATConv(
                        hid_dim,
                        hid_dim // num_heads,
                        heads=num_heads,
                        dropout=dropout,
                        edge_dim=edge_dim if use_edge_features else None,
                        concat=True,
                    )
                elif self.gnn_type == "graphsage":
                    conv = gnn.SAGEConv(hid_dim, hid_dim)
                else:
                    raise ValueError(
                        f"Unknown gnn_type: {self.gnn_type}. "
                        f"Supported types: gcn, gat, graphsage, gin, transformer."
                    )
                self.convs.append(conv)
                self.batch_norms.append(nn.BatchNorm1d(hid_dim))

            if pool_type == "mean":
                self.readout = lambda x, batch, edge_index, edge_attr: global_mean_pool(
                    x, batch
                )
            elif pool_type == "sum":
                self.readout = lambda x, batch, edge_index, edge_attr: global_add_pool(
                    x, batch
                )
            elif pool_type == "max":
                self.readout = lambda x, batch, edge_index, edge_attr: global_max_pool(
                    x, batch
                )
            elif pool_type == "transformer":
                self.readout = TransformerPool(
                    in_channels=hid_dim,
                    num_heads=pool_num_heads,
                    dim_feedforward=pool_dim_feedforward,
                    dropout_rate=dropout,
                )
            elif pool_type == "sag":
                self.readout = SAGPool(in_channels=hid_dim, ratio=pool_ratio)
            elif pool_type == "topk":
                self.readout = TopKPool(in_channels=hid_dim, ratio=pool_ratio)
            else:
                raise ValueError(f"Unknown pool_type: {pool_type}")

            mlp_layers = []
            for i in range(ffn_num_layers):
                if i == 0:
                    mlp_layers.append(nn.Linear(hid_dim, ffn_hidden_dim))
                else:
                    mlp_layers.append(nn.Linear(ffn_hidden_dim, ffn_hidden_dim))
                mlp_layers.append(get_activation(activation))
                mlp_layers.append(nn.Dropout(dropout))
            mlp_layers.append(nn.Linear(ffn_hidden_dim, num_targets))
            self.out_mlp = nn.Sequential(*mlp_layers)

        def forward(self, data):
            x, edge_index, edge_attr, batch = (
                data.x,
                data.edge_index,
                data.edge_attr,
                data.batch,
            )
            x = x.float()
            if edge_attr is not None:
                edge_attr = edge_attr.float()
            x = self.node_encoder(x)
            edge_attr_encoded = (
                self.edge_encoder(edge_attr) if self.edge_encoder is not None else None
            )
            for i, conv in enumerate(self.convs):
                x_prev = x
                if self.gnn_type == "gcn":
                    x = conv(x, edge_index)
                elif self.gnn_type == "gin":
                    x = conv(x, edge_index)
                elif self.gnn_type == "transformer":
                    x = conv(x, edge_index, edge_attr_encoded)
                elif self.gnn_type == "gat":
                    if self.use_edge_features and edge_attr is not None:
                        x = conv(x, edge_index, edge_attr=edge_attr)
                    else:
                        x = conv(x, edge_index)
                elif self.gnn_type == "graphsage":
                    x = conv(x, edge_index)
                x = self.batch_norms[i](x)
                x = self.activation_fn(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if i > 0:
                    x = x + x_prev
            if self.pool_type in ["mean", "sum", "max"]:
                graph_feat = self.readout(x, batch, edge_index, edge_attr_encoded)
            else:
                graph_feat = self.readout(x, edge_index, batch, edge_attr_encoded)
            out = self.out_mlp(graph_feat)
            return out

    return (GenericPyGModel,)


@app.cell
def _deep_gcn(
    AttentiveFPReadout,
    F,
    MessagePassing,
    SAGPool,
    TopKPool,
    TransformerPool,
    get_activation,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
    nn,
    softmax,
    torch,
):
    # ----------------------- DeeperGCN model --------------------------------
    class GENConv(MessagePassing):
        def __init__(
            self,
            in_dim,
            out_dim,
            aggregator="softmax",
            beta=1.0,
            learn_beta=True,
            mlp_layers=1,
            norm="layer",
            activation="relu",
        ):
            super().__init__(aggr=None)
            self.in_dim = in_dim
            self.out_dim = out_dim
            self.aggregator = aggregator
            if learn_beta:
                self.beta = nn.Parameter(torch.tensor(beta))
            else:
                self.register_buffer("beta", torch.tensor(beta))
            act_fn = get_activation(activation)
            mlp = []
            for i in range(mlp_layers):
                if i == 0:
                    mlp.append(nn.Linear(in_dim, out_dim))
                else:
                    mlp.append(nn.Linear(out_dim, out_dim))
                if norm == "batch":
                    mlp.append(nn.BatchNorm1d(out_dim))
                elif norm == "layer":
                    mlp.append(nn.LayerNorm(out_dim))
                elif norm == "instance":
                    mlp.append(nn.InstanceNorm1d(out_dim))
                mlp.append(get_activation(activation))
            self.msg_norm = nn.Sequential(*mlp)
            self.edge_encoder = nn.Linear(in_dim, out_dim)

        def forward(self, x, edge_index, edge_attr):
            edge_embedding = self.edge_encoder(edge_attr)
            return self.propagate(edge_index, x=x, edge_attr=edge_embedding)

        def message(self, x_j, edge_attr):
            return self.msg_norm(x_j + edge_attr)

        def aggregate(self, inputs, index, dim_size=None):
            if self.aggregator == "softmax":
                out = softmax(inputs * self.beta, index, dim=self.node_dim)
                out = out * inputs
                if dim_size is None:
                    dim_size = int(index.max()) + 1
                output = torch.zeros(
                    dim_size, inputs.size(-1), dtype=inputs.dtype, device=inputs.device
                )
                return output.scatter_add_(
                    0, index.unsqueeze(-1).expand_as(inputs), out
                )
            elif self.aggregator == "power":
                min_value, max_value = 1e-7, 1e1
                torch.clamp_(inputs, min_value, max_value)
                if dim_size is None:
                    dim_size = int(index.max()) + 1
                output = torch.zeros(
                    dim_size, inputs.size(-1), dtype=inputs.dtype, device=inputs.device
                )
                output = output.scatter_add_(
                    0, index.unsqueeze(-1).expand_as(inputs), inputs.pow(self.beta)
                )
                return output.clamp(min_value, max_value).pow(1.0 / self.beta)
            else:
                raise ValueError(f"Unknown aggregator: {self.aggregator}")

    class DeeperGCN(nn.Module):
        def __init__(
            self,
            node_in_dim,
            edge_in_dim,
            hid_dim,
            num_layers,
            dropout=0.0,
            activation="relu",
            norm="layer",
            beta=1.0,
            learn_beta=True,
            aggr="softmax",
            mlp_layers=1,
            pool_type="mean",
            pool_ratio=0.5,
            pool_num_heads=4,
            pool_dim_feedforward=128,
            pool_num_timesteps=2,
            ffn_hidden_dim=300,
            ffn_num_layers=2,
            num_targets=1,
        ):
            super().__init__()
            self.num_layers = num_layers
            self.dropout = dropout
            self.pool_type = pool_type
            self.activation_fn = get_activation(activation)

            self.node_encoder = nn.Linear(node_in_dim, hid_dim)
            self.edge_encoder = nn.Linear(edge_in_dim, hid_dim)

            self.gcns = nn.ModuleList()
            self.norms = nn.ModuleList()
            for _ in range(num_layers):
                conv = GENConv(
                    in_dim=hid_dim,
                    out_dim=hid_dim,
                    aggregator=aggr,
                    beta=beta,
                    learn_beta=learn_beta,
                    mlp_layers=mlp_layers,
                    norm=norm,
                    activation=activation,
                )
                self.gcns.append(conv)
                if norm == "batch":
                    self.norms.append(nn.BatchNorm1d(hid_dim))
                elif norm == "layer":
                    self.norms.append(nn.LayerNorm(hid_dim))
                elif norm == "instance":
                    self.norms.append(nn.InstanceNorm1d(hid_dim))
                else:
                    raise ValueError(f"Unknown norm type: {norm}")

            if pool_type == "mean":
                self.readout = lambda x, batch, edge_index, edge_attr: global_mean_pool(
                    x, batch
                )
            elif pool_type == "sum":
                self.readout = lambda x, batch, edge_index, edge_attr: global_add_pool(
                    x, batch
                )
            elif pool_type == "max":
                self.readout = lambda x, batch, edge_index, edge_attr: global_max_pool(
                    x, batch
                )
            elif pool_type == "transformer":
                self.readout = TransformerPool(
                    in_channels=hid_dim,
                    num_heads=pool_num_heads,
                    dim_feedforward=pool_dim_feedforward,
                    dropout_rate=dropout,
                )
            elif pool_type == "sag":
                self.readout = SAGPool(in_channels=hid_dim, ratio=pool_ratio)
            elif pool_type == "topk":
                self.readout = TopKPool(in_channels=hid_dim, ratio=pool_ratio)
            elif pool_type == "attentivefp":
                self.readout = AttentiveFPReadout(
                    feat_size=hid_dim, num_timesteps=pool_num_timesteps, dropout=dropout
                )
            else:
                raise ValueError(f"Unknown pool_type: {pool_type}")

            mlp_list = []
            for i in range(ffn_num_layers):
                if i == 0:
                    mlp_list.append(nn.Linear(hid_dim, ffn_hidden_dim))
                else:
                    mlp_list.append(nn.Linear(ffn_hidden_dim, ffn_hidden_dim))
                mlp_list.append(get_activation(activation))
                mlp_list.append(nn.Dropout(dropout))
            mlp_list.append(nn.Linear(ffn_hidden_dim, num_targets))
            self.out_mlp = nn.Sequential(*mlp_list)

        def forward(self, data):
            x, edge_index, edge_attr, batch = (
                data.x,
                data.edge_index,
                data.edge_attr,
                data.batch,
            )
            x = x.float()
            if edge_attr is not None:
                edge_attr = edge_attr.float()
            x = self.node_encoder(x)
            edge_attr = self.edge_encoder(edge_attr)
            for layer in range(self.num_layers):
                h = self.norms[layer](x)
                h = self.activation_fn(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                x = self.gcns[layer](h, edge_index, edge_attr) + x
            if self.pool_type in ["mean", "sum", "max"]:
                graph_feat = self.readout(x, batch, edge_index, edge_attr)
            elif self.pool_type == "attentivefp":
                graph_feat = self.readout(x, batch)
            else:
                graph_feat = self.readout(x, edge_index, batch, edge_attr)
            out = self.out_mlp(graph_feat)
            return out

    return (DeeperGCN,)


@app.cell
def _build_model(DeeperGCN, GenericPyGModel):
    # ----------------------- Model factory (PyG only) -----------------------
    def build_model(model_config):
        pyg_cfg = model_config.pyg
        activation = pyg_cfg.activation.lower()
        if activation not in ["relu", "silu", "gelu"]:
            raise ValueError(
                f"Unsupported activation: '{activation}'. "
                f"Must be one of: 'relu', 'silu', 'gelu'"
            )
        edge_dim = pyg_cfg.edge_dim if pyg_cfg.edge_dim is not None else pyg_cfg.edge_in_dim
        if pyg_cfg.gnn_type == "deepgcn":
            return DeeperGCN(
                node_in_dim=pyg_cfg.node_in_dim,
                edge_in_dim=pyg_cfg.edge_in_dim,
                hid_dim=model_config.message_hidden_dim,
                num_layers=model_config.num_layers,
                dropout=model_config.dropout,
                activation=activation,
                norm=pyg_cfg.deepgcn.norm_type,
                beta=pyg_cfg.deepgcn.beta,
                learn_beta=pyg_cfg.deepgcn.learn_beta,
                aggr=pyg_cfg.deepgcn.gen_aggr,
                mlp_layers=pyg_cfg.deepgcn.mlp_layers,
                pool_type=pyg_cfg.pool_type,
                pool_ratio=pyg_cfg.pool_ratio,
                pool_num_heads=pyg_cfg.pool_num_heads,
                pool_dim_feedforward=pyg_cfg.pool_dim_feedforward,
                pool_num_timesteps=pyg_cfg.pool_num_timesteps,
                ffn_hidden_dim=model_config.ffn_hidden_dim,
                ffn_num_layers=model_config.ffn_num_layers,
                num_targets=model_config.num_targets,
            )
        return GenericPyGModel(
            node_in_dim=pyg_cfg.node_in_dim,
            edge_in_dim=pyg_cfg.edge_in_dim,
            hid_dim=model_config.message_hidden_dim,
            num_layers=model_config.num_layers,
            gnn_type=pyg_cfg.gnn_type,
            dropout=model_config.dropout,
            activation=activation,
            pool_type=pyg_cfg.pool_type,
            pool_ratio=pyg_cfg.pool_ratio,
            pool_num_heads=pyg_cfg.pool_num_heads,
            pool_dim_feedforward=pyg_cfg.pool_dim_feedforward,
            num_heads=pyg_cfg.num_heads,
            edge_dim=edge_dim if pyg_cfg.gnn_type in ("transformer", "gat") else None,
            ffn_hidden_dim=model_config.ffn_hidden_dim,
            ffn_num_layers=model_config.ffn_num_layers,
            num_targets=model_config.num_targets,
            use_edge_features=pyg_cfg.use_edge_features,
        )

    def count_model_params(model_config):
        m = build_model(model_config)
        return sum(p.numel() for p in m.parameters())

    return build_model, count_model_params


@app.cell
def _trainer(
    CSVLogger,
    EarlyStopping,
    L,
    LearningRateMonitor,
    ModelCheckpoint,
    Path,
    RTDataModule,
    build_model,
    json,
    math,
    nn,
    torch,
):
    class GradientClippingCallback(L.Callback):
        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            grad_norm = torch.nn.utils.clip_grad_norm_(
                pl_module.parameters(), max_norm=float("inf")
            )
            pl_module.log("train/grad_norm", grad_norm, prog_bar=False)

    class CosineAnnealingWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
        def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=1e-7, last_epoch=-1):
            self.warmup_epochs = warmup_epochs
            self.max_epochs = max_epochs
            self.eta_min = eta_min
            super().__init__(optimizer, last_epoch)

        def get_lr(self):
            if self.last_epoch < self.warmup_epochs:
                return [
                    base_lr * (self.last_epoch + 1) / self.warmup_epochs
                    for base_lr in self.base_lrs
                ]
            progress = (self.last_epoch - self.warmup_epochs) / (
                self.max_epochs - self.warmup_epochs
            )
            return [
                self.eta_min
                + (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2
                for base_lr in self.base_lrs
            ]

    class RTTrainer(L.LightningModule):
        """PyG-only Lightning module wrapping a GNN for multi-target regression."""

        def __init__(self, model, training_config, target_means, target_stds):
            super().__init__()
            self.model = model
            self.training_config = training_config
            self.target_means = dict(target_means)
            self.target_stds = dict(target_stds)
            self.num_targets = len(target_means)
            if self.num_targets == 0:
                raise ValueError("target_means must be non-empty")
            if set(target_means.keys()) != set(target_stds.keys()):
                raise ValueError("target_means and target_stds keys must match")
            target_keys = list(target_means.keys())
            self.register_buffer(
                "target_means_tensor",
                torch.tensor(
                    [target_means[k] for k in target_keys], dtype=torch.float32
                ),
            )
            self.register_buffer(
                "target_stds_tensor",
                torch.tensor(
                    [target_stds[k] for k in target_keys], dtype=torch.float32
                ),
            )

            if training_config.loss_fn == "mse":
                self.loss_fn = nn.MSELoss()
            elif training_config.loss_fn == "mae":
                self.loss_fn = nn.L1Loss()
            elif training_config.loss_fn == "huber":
                self.loss_fn = nn.HuberLoss(delta=training_config.huber_delta)
            elif training_config.loss_fn == "smooth_l1":
                self.loss_fn = nn.SmoothL1Loss(beta=training_config.huber_delta)
            else:
                raise ValueError(f"Unknown loss function: {training_config.loss_fn}")
            print(f"[RTTrainer] Using loss function: {training_config.loss_fn}")
            print(
                f"[RTTrainer] Number of targets: {self.num_targets} "
                f"({', '.join(target_keys)})"
            )
            self.best_val_loss = float("inf")
            self.val_loss_spike_count = 0
            self.save_hyperparameters(ignore=["model"])

        def forward(self, batch):
            return self.model(batch)

        def _shared_step(self, batch, batch_idx, stage):
            preds = self(batch)
            targets = batch.y
            batch_size = int(batch.batch.max().item()) + 1
            preds = preds.reshape(batch_size, self.num_targets)
            targets = targets.reshape(batch_size, self.num_targets)
            if hasattr(batch, "y_mask") and batch.y_mask is not None:
                mask = batch.y_mask.reshape(batch_size, self.num_targets).bool()
            else:
                mask = ~torch.isnan(targets)

            total_loss = torch.zeros((), device=preds.device, dtype=preds.dtype)
            for j in range(self.num_targets):
                target_mask = mask[:, j]
                if not target_mask.any():
                    continue
                preds_j = preds[:, j][target_mask]
                targets_j = targets[:, j][target_mask]
                target_loss = self.loss_fn(preds_j, targets_j)
                if not (torch.isnan(target_loss) or torch.isinf(target_loss)):
                    total_loss = total_loss + target_loss

            if stage == "train" and (torch.isnan(total_loss) or torch.isinf(total_loss)):
                self.log("train/nan_loss", 1.0, batch_size=batch_size)
                return torch.tensor(0.0, requires_grad=True, device=preds.device)

            preds_denorm = preds * self.target_stds_tensor + self.target_means_tensor
            targets_denorm = targets * self.target_stds_tensor + self.target_means_tensor
            target_names = list(self.target_means.keys())
            maes, rmses, r2s = [], [], []
            for j in range(self.num_targets):
                target_mask = mask[:, j]
                if not target_mask.any():
                    continue
                p_j = preds_denorm[:, j][target_mask]
                t_j = targets_denorm[:, j][target_mask]
                diff = p_j - t_j
                mae_j = torch.abs(diff).mean()
                rmse_j = torch.sqrt(torch.pow(diff, 2).mean())
                ss_res = torch.sum(diff ** 2)
                ss_tot = torch.sum((t_j - t_j.mean()) ** 2)
                r2_j = 1 - ss_res / (ss_tot + 1e-8)
                name = target_names[j]
                self.log(
                    f"{stage}/mae_{name}", mae_j,
                    on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size,
                )
                self.log(
                    f"{stage}/rmse_{name}", rmse_j,
                    on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size,
                )
                self.log(
                    f"{stage}/r2_{name}", r2_j,
                    on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size,
                )
                maes.append(mae_j)
                rmses.append(rmse_j)
                r2s.append(r2_j)

            if maes:
                mae_mean = torch.stack(maes).mean()
                rmse_mean = torch.stack(rmses).mean()
                r2_mean = torch.stack(r2s).mean()
            else:
                zero = torch.zeros((), device=preds.device, dtype=preds.dtype)
                mae_mean = rmse_mean = r2_mean = zero

            self.log(
                f"{stage}/loss", total_loss,
                prog_bar=True, on_step=False, on_epoch=True,
                sync_dist=True, batch_size=batch_size,
            )
            self.log(
                f"{stage}/mae", mae_mean,
                prog_bar=(stage != "train"), on_step=False, on_epoch=True,
                sync_dist=True, batch_size=batch_size,
            )
            self.log(
                f"{stage}/mae_mean", mae_mean,
                on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size,
            )
            self.log(
                f"{stage}/rmse", rmse_mean,
                on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size,
            )
            self.log(
                f"{stage}/rmse_mean", rmse_mean,
                on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size,
            )
            self.log(
                f"{stage}/r2", r2_mean,
                on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size,
            )
            self.log(
                f"{stage}/r2_mean", r2_mean,
                on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size,
            )
            return total_loss

        def training_step(self, batch, batch_idx):
            return self._shared_step(batch, batch_idx, "train")

        def validation_step(self, batch, batch_idx):
            return self._shared_step(batch, batch_idx, "val")

        def test_step(self, batch, batch_idx):
            return self._shared_step(batch, batch_idx, "test")

        def on_validation_epoch_end(self):
            val_loss = self.trainer.callback_metrics.get("val/loss")
            val_mae = self.trainer.callback_metrics.get("val/mae")
            val_rmse = self.trainer.callback_metrics.get("val/rmse")
            val_r2 = self.trainer.callback_metrics.get("val/r2")
            current_epoch = (
                self.current_epoch if hasattr(self, "current_epoch")
                else self.trainer.current_epoch
            )
            print(
                f"\nEpoch {current_epoch}: "
                f"val/loss={val_loss} val/mae={val_mae} "
                f"val/rmse={val_rmse} val/r2={val_r2}"
            )
            if val_loss is not None:
                if val_loss > self.best_val_loss * 3.0:
                    self.val_loss_spike_count += 1
                    self.log("val/loss_spike", 1.0)
                    print(
                        f"\nWARNING: Validation loss spike detected! "
                        f"Previous best: {self.best_val_loss:.2f}, "
                        f"Current: {val_loss:.2f}"
                    )
                else:
                    self.val_loss_spike_count = 0
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss.item()

        def configure_optimizers(self):
            cfg = self.training_config
            if cfg.optimizer == "adam":
                optimizer = torch.optim.Adam(
                    self.parameters(), lr=cfg.learning_rate,
                    weight_decay=cfg.weight_decay, eps=1e-8,
                )
            elif cfg.optimizer == "adamw":
                optimizer = torch.optim.AdamW(
                    self.parameters(), lr=cfg.learning_rate,
                    weight_decay=cfg.weight_decay, eps=1e-8,
                )
            elif cfg.optimizer == "sgd":
                optimizer = torch.optim.SGD(
                    self.parameters(), lr=cfg.learning_rate, momentum=0.9,
                    weight_decay=cfg.weight_decay,
                )
            else:
                raise ValueError(f"Unknown optimizer: {cfg.optimizer}")
            if not cfg.use_scheduler:
                return optimizer
            if cfg.scheduler_type == "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode=cfg.monitor_mode,
                    patience=cfg.scheduler_patience, factor=cfg.scheduler_factor,
                    min_lr=1e-7,
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "monitor": cfg.monitor_metric},
                }
            elif cfg.scheduler_type == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=cfg.num_epochs, eta_min=1e-7
                )
            elif cfg.scheduler_type == "cosine_warmup":
                scheduler = CosineAnnealingWarmupScheduler(
                    optimizer, warmup_epochs=cfg.warmup_epochs,
                    max_epochs=cfg.num_epochs, eta_min=1e-7,
                )
            elif cfg.scheduler_type == "step":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=cfg.scheduler_patience, gamma=cfg.scheduler_factor,
                )
            else:
                raise ValueError(f"Unknown scheduler: {cfg.scheduler_type}")
            return [optimizer], [scheduler]

    def train_from_config(config, num_workers=0):
        print("[train_from_config] Starting training pipeline")
        L.seed_everything(config.training.seed, workers=True)
        datamodule = RTDataModule(
            config=config.data,
            batch_size=config.training.batch_size,
            num_workers=num_workers,
        )
        datamodule.prepare_data()
        datamodule.setup()
        config.model.num_targets = len(datamodule.target_columns)
        print(
            f"[train_from_config] Using num_targets={config.model.num_targets} "
            f"({', '.join(datamodule.target_columns)})"
        )
        model = build_model(config.model)
        module = RTTrainer(
            model=model,
            training_config=config.training,
            target_means=datamodule.target_means,
            target_stds=datamodule.target_stds,
        )

        checkpoint_dir = (
            config.training.checkpoint_dir
            / config.data.dataset_name
            / config.experiment_name
        )
        log_dir = config.training.log_dir / config.data.dataset_name

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best_model",
            monitor=config.training.monitor_metric,
            mode=config.training.monitor_mode,
            save_top_k=config.training.save_top_k,
            save_last=True,
        )
        early_stop_callback = EarlyStopping(
            monitor=config.training.monitor_metric,
            patience=config.training.early_stop_patience,
            mode=config.training.monitor_mode,
            verbose=True,
            check_finite=True,
        )
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        grad_monitor = GradientClippingCallback()

        logger = CSVLogger(save_dir=log_dir, name=config.experiment_name)
        config_path = Path(logger.log_dir) / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2, default=str)
        print(f"[train_from_config] Saved config to {config_path}")

        devices_param = config.training.devices
        if isinstance(devices_param, str):
            if "," in devices_param:
                devices_param = [int(d.strip()) for d in devices_param.split(",")]
            else:
                try:
                    devices_param = int(devices_param)
                except ValueError:
                    devices_param = 1

        trainer = L.Trainer(
            max_epochs=config.training.num_epochs,
            accelerator=config.training.accelerator,
            devices=devices_param,
            precision=config.training.precision,
            callbacks=[checkpoint_callback, early_stop_callback, lr_monitor, grad_monitor],
            logger=logger,
            log_every_n_steps=config.training.log_every_n_steps,
            deterministic=config.training.deterministic,
            enable_progress_bar=False,
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm",
            detect_anomaly=False,
            accumulate_grad_batches=1,
        )
        print("[train_from_config] Starting training...")
        trainer.fit(module, datamodule=datamodule)
        print("[train_from_config] Testing with best checkpoint...")
        test_results = None
        if checkpoint_callback.best_model_path:
            print(
                f"[train_from_config] Loading best model from "
                f"{checkpoint_callback.best_model_path}"
            )
            ckpt = torch.load(
                checkpoint_callback.best_model_path, map_location="cpu", weights_only=False,
            )
            module.load_state_dict(ckpt["state_dict"])
            test_results = trainer.test(module, datamodule=datamodule)
        else:
            print("[train_from_config] No best checkpoint found, testing with final model")
            test_results = trainer.test(module, datamodule=datamodule)
        print("[train_from_config] Training complete!")
        return trainer, module, datamodule, test_results

    return (train_from_config,)


@app.cell
def _helpers(copy, count_model_params, pl):
    # ------------------ Pivot / row helpers (shared) ------------------------

    def build_size_result_row(
        split_method,
        depth,
        width,
        num_params,
        test_results,
    ):
        row = {
            "split_method": split_method,
            "depth": depth,
            "width": width,
            "num_params": num_params,
        }
        if test_results:
            row["test/mae"] = test_results.get(
                "test/mae_mean", test_results.get("test/mae", float("nan"))
            )
            row["test/rmse"] = test_results.get(
                "test/rmse_mean", test_results.get("test/rmse", float("nan"))
            )
            row["test/r2"] = test_results.get(
                "test/r2_mean", test_results.get("test/r2", float("nan"))
            )
            for key, value in test_results.items():
                if key.startswith("test/") and key not in row:
                    row[key] = value
        else:
            row["test/mae"] = float("nan")
            row["test/rmse"] = float("nan")
            row["test/r2"] = float("nan")
        return row

    def build_arch_result_row(
        split_method,
        arch,
        depth,
        requested_width,
        chosen_width,
        num_params,
        test_results,
    ):
        row = {
            "split_method": split_method,
            "architecture": arch,
            "depth": depth,
            "requested_width": requested_width,
            "chosen_width": chosen_width,
            "num_params": num_params,
        }
        if test_results:
            row["test/mae"] = test_results.get(
                "test/mae_mean", test_results.get("test/mae", float("nan"))
            )
            row["test/rmse"] = test_results.get(
                "test/rmse_mean", test_results.get("test/rmse", float("nan"))
            )
            row["test/r2"] = test_results.get(
                "test/r2_mean", test_results.get("test/r2", float("nan"))
            )
            for key, value in test_results.items():
                if key.startswith("test/") and key not in row:
                    row[key] = value
        else:
            row["test/mae"] = float("nan")
            row["test/rmse"] = float("nan")
            row["test/r2"] = float("nan")
        return row

    def choose_architecture_width(
        arch, depth, width, num_heads, base_model_cfg, num_targets,
    ):
        ref_cfg = copy.deepcopy(base_model_cfg)
        ref_cfg.model_type = "pyg"
        ref_cfg.pyg.gnn_type = "gcn"
        ref_cfg.pyg.pool_type = "mean"
        ref_cfg.num_layers = depth
        ref_cfg.message_hidden_dim = width
        ref_cfg.num_targets = num_targets
        reference_params = count_model_params(ref_cfg)

        if arch == "gcn":
            return width, reference_params

        step = num_heads if arch in ("gat", "transformer") else 1
        low = max(step, 16)
        high = 8192

        best_width = width
        best_diff = float("inf")

        def _params_for(w):
            cfg = copy.deepcopy(base_model_cfg)
            cfg.model_type = "pyg"
            cfg.pyg.gnn_type = arch
            cfg.pyg.pool_type = "mean"
            cfg.num_layers = depth
            cfg.message_hidden_dim = w
            cfg.num_targets = num_targets
            return count_model_params(cfg)

        lo, hi = low, high
        while lo <= hi:
            mid = (lo + hi) // 2
            mid = (mid // step) * step
            mid = max(mid, lo)
            params = _params_for(mid)
            diff = abs(params - reference_params)
            if diff < best_diff:
                best_diff = diff
                best_width = mid
            if params < reference_params:
                lo = mid + step
            else:
                hi = mid - step

        candidates = {width}
        for w in range(
            max(low, best_width - 4 * step),
            min(high, best_width + 4 * step) + 1,
            step,
        ):
            candidates.add(w)
        for w in candidates:
            params = _params_for(w)
            diff = abs(params - reference_params)
            if diff < best_diff:
                best_diff = diff
                best_width = w
        return best_width, reference_params

    def _print_pivot(df, index_col, on_col, value_col, split_methods):
        if df.is_empty() or value_col not in df.columns:
            print(f"\n--- {value_col} --- (no data)")
            return
        pivot = df.pivot(index=index_col, on=on_col, values=value_col)
        for method in split_methods:
            if method not in pivot.columns:
                pivot = pivot.with_columns(pl.lit(None).alias(method))
        pivot = pivot.select([index_col] + split_methods)
        print(f"\n--- {value_col} ---")
        header = "| " + " | ".join(pivot.columns) + " |"
        separator = "|---" * len(pivot.columns) + "|"
        print(header)
        print(separator)
        for row_data in pivot.iter_rows():
            cells = []
            for val in row_data:
                if val is None:
                    cells.append("NaN")
                elif isinstance(val, float):
                    cells.append(f"{val:.4f}")
                else:
                    cells.append(str(val))
            print("| " + " | ".join(cells) + " |")

    return (
        build_arch_result_row,
        build_size_result_row,
        choose_architecture_width,
    )


@app.cell
def _size_comparison(
    CSV_PATH,
    Config,
    DataConfig,
    ModelConfig,
    PyGModelConfig,
    SIZE_RESULTS_CSV,
    TrainingConfig,
    build_size_result_row,
    copy,
    pl,
    product,
    train_from_config,
):
    # ------------------ Systematic size comparison (PyG only) ----------------

    def systematic_size_comparison(
        split_methods=None,
        depths=None,
        widths=None,
        num_epochs=30,
        batch_size=128,
        learning_rate=1e-3,
        gnn_type="deepgcn",
        num_workers=0,
    ):
        """Run the systematic size comparison across split methods and depths/widths.

        Resumable: reads ``size_results_csv`` and skips experiments already
        present. Writes incrementally after each run.
        """
        if split_methods is None:
            split_methods = ["random", "scaffold", "butina", "mces", "mces_umap"]
        if depths is None:
            depths = [4, 8]
        if widths is None:
            widths = [128, 256, 512]

        # Base data config tuned for enveda_180.csv.
        base_data_config = DataConfig(
            raw_data_path=CSV_PATH,
            cid_column="name",
            target_columns=["rt_seconds", "ccs"],
            inchi_column="smiles",
            smiles_column="smiles",
            dataset_name="enveda_180",
            filter_invalid_inchi=False,
            target_filters={},
            featurizer_type="rdkit",
        )
        base_model_config = ModelConfig(
            model_type="pyg",
            message_hidden_dim=widths[0],
            num_layers=depths[0],
            ffn_hidden_dim=widths[0],
            ffn_num_layers=2,
            dropout=0.1,
            num_targets=2,
            pyg=PyGModelConfig(
                gnn_type=gnn_type,
                pool_type="mean",
                activation="relu",
                num_heads=4,
                use_edge_features=True,
            ),
        )
        base_training_config = TrainingConfig(
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=num_epochs,
            optimizer="adam",
            weight_decay=0.0,
            loss_fn="mse",
            monitor_metric="val/loss",
            monitor_mode="min",
            early_stop_patience=20,
            accelerator="auto",
            devices="auto",
            precision="32",
            seed=42,
            deterministic=False,
        )

        if SIZE_RESULTS_CSV.exists():
            print(f"Loading existing size results from {SIZE_RESULTS_CSV}")
            existing_df = pl.read_csv(SIZE_RESULTS_CSV)
            results = existing_df.to_dicts()
            completed = {
                (row["split_method"], int(row["depth"]), int(row["width"]))
                for row in results
            }
        else:
            results = []
            completed = set()

        for split_method in split_methods:
            for depth, width in product(depths, widths):
                if (split_method, depth, width) in completed:
                    print(
                        f"Skipping already completed: split={split_method}, "
                        f"depth={depth}, width={width}"
                    )
                    continue
                print(
                    f"\nRunning experiment: split={split_method}, "
                    f"depth={depth}, width={width}"
                )
                data_cfg = copy.deepcopy(base_data_config)
                data_cfg.split_method = split_method
                model_cfg = copy.deepcopy(base_model_config)
                model_cfg.num_layers = depth
                model_cfg.message_hidden_dim = width
                model_cfg.ffn_hidden_dim = width
                config = Config(
                    data=data_cfg,
                    model=model_cfg,
                    training=copy.deepcopy(base_training_config),
                    experiment_name=(
                        f"size_comp_{split_method}_d{depth}_w{width}"
                    ),
                    description=(
                        f"Size comparison: {gnn_type} on {split_method} split "
                        f"with depth {depth} and width {width}"
                    ),
                    tags=["size_comparison", gnn_type],
                )
                try:
                    _, module, _, test_results_list = train_from_config(
                        config, num_workers=num_workers
                    )
                    test_results = test_results_list[0] if test_results_list else None
                    num_params = sum(p.numel() for p in module.model.parameters())
                except Exception as e:
                    print(
                        f"--- Experiment failed for split: {split_method}, "
                        f"depth: {depth}, width: {width} ---"
                    )
                    print(e)
                    test_results = None
                    num_params = 0
                row = build_size_result_row(
                    split_method, depth, width, num_params, test_results
                )
                results.append(row)
                completed.add((split_method, depth, width))
                df = pl.DataFrame(results)
                df.write_csv(SIZE_RESULTS_CSV)
                print(f"Saved results to {SIZE_RESULTS_CSV}")

        if not results:
            print("No results were generated.")
            return pl.DataFrame()
        df = pl.DataFrame(results).sort("num_params")
        df = df.with_columns(
            pl.concat_str(
                pl.lit("d="),
                pl.col("depth").cast(pl.String),
                pl.lit(", w="),
                pl.col("width").cast(pl.String),
                pl.lit(" ("),
                pl.col("num_params").cast(pl.String),
                pl.lit(")"),
            ).alias("size_label")
        )
        for metric in ["test/mae", "test/rmse", "test/r2"]:
            pivot = df.pivot(
                index="size_label", on="split_method", values=metric,
            )
            for method in split_methods:
                if method not in pivot.columns:
                    pivot = pivot.with_columns(pl.lit(None).alias(method))
            pivot = pivot.select(["size_label"] + split_methods)
            print(f"\n--- Size comparison results for {metric} ---")
            header = "| " + " | ".join(pivot.columns) + " |"
            separator = "|---" * len(pivot.columns) + "|"
            print(header)
            print(separator)
            for row_data in pivot.iter_rows():
                cells = []
                for val in row_data:
                    if val is None:
                        cells.append("NaN")
                    elif isinstance(val, float):
                        cells.append(f"{val:.4f}")
                    else:
                        cells.append(str(val))
                print("| " + " | ".join(cells) + " |")
        return df

    return (systematic_size_comparison,)


@app.cell
def _arch_comparison(
    ARCH_RESULTS_CSV,
    CSV_PATH,
    Config,
    DataConfig,
    ModelConfig,
    PyGModelConfig,
    RTDataModule,
    TrainingConfig,
    build_arch_result_row,
    choose_architecture_width,
    copy,
    count_model_params,
    pl,
    train_from_config,
):
    # ------------ Systematic architecture comparison (PyG only) -------------

    def systematic_architecture_comparison(
        split_methods=None,
        architectures=None,
        depth=4,
        width=256,
        size_mode="normalized_params",
        num_epochs=30,
        batch_size=128,
        learning_rate=1e-3,
        num_workers=0,
    ):
        """Run the systematic architecture comparison.

        Resumable: reads ``arch_results_csv`` and skips experiments already
        present. Writes incrementally after each run.
        """
        if split_methods is None:
            split_methods = ["random", "scaffold", "butina", "mces", "mces_umap"]
        if architectures is None:
            architectures = {
                "gcn": True,
                "gat": True,
                "graphsage": True,
                "gin": True,
                "deepgcn": True,
                "transformer": True,
            }

        base_data_config = DataConfig(
            raw_data_path=CSV_PATH,
            cid_column="name",
            target_columns=["rt_seconds", "ccs"],
            inchi_column="smiles",
            smiles_column="smiles",
            dataset_name="enveda_180",
            filter_invalid_inchi=False,
            target_filters={},
            featurizer_type="rdkit",
        )
        base_model_config = ModelConfig(
            model_type="pyg",
            message_hidden_dim=width,
            num_layers=depth,
            ffn_hidden_dim=width,
            ffn_num_layers=2,
            dropout=0.1,
            num_targets=2,
            pyg=PyGModelConfig(
                gnn_type="gcn",
                pool_type="mean",
                activation="relu",
                num_heads=4,
                use_edge_features=True,
            ),
        )
        base_training_config = TrainingConfig(
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=num_epochs,
            optimizer="adam",
            weight_decay=0.0,
            loss_fn="mse",
            monitor_metric="val/loss",
            monitor_mode="min",
            early_stop_patience=20,
            accelerator="auto",
            devices="auto",
            precision="32",
            seed=42,
            deterministic=False,
        )

        num_targets = len(base_data_config.target_columns)
        num_heads = base_model_config.pyg.num_heads

        if ARCH_RESULTS_CSV.exists():
            print(f"Loading existing arch results from {ARCH_RESULTS_CSV}")
            existing_df = pl.read_csv(ARCH_RESULTS_CSV)
            results = existing_df.to_dicts()
            completed = {
                (
                    row["split_method"],
                    row["architecture"],
                    int(row["depth"]),
                    int(row["requested_width"]),
                )
                for row in results
            }
        else:
            results = []
            completed = set()

        effective_batch_size = batch_size

        for split_method in split_methods:
            print(f"\n===== Split method: {split_method} =====")
            data_cfg = copy.deepcopy(base_data_config)
            data_cfg.split_method = split_method
            dm = RTDataModule(
                config=data_cfg,
                batch_size=effective_batch_size,
                num_workers=num_workers,
            )
            print(f"[ArchitectureComparison] Preparing split for {split_method}...")
            dm.prepare_data()

            for arch, enabled in architectures.items():
                if not enabled:
                    continue
                key = (split_method, arch, depth, width)
                if key in completed:
                    print(f"Skipping completed experiment: {key}")
                    continue
                print(f"\n[ArchitectureComparison] Running architecture: {arch}")
                model_cfg = copy.deepcopy(base_model_config)
                model_cfg.model_type = "pyg"
                model_cfg.pyg.gnn_type = arch
                model_cfg.num_layers = depth
                model_cfg.num_targets = num_targets
                if arch != "deepgcn":
                    model_cfg.pyg.pool_type = "mean"

                if size_mode == "fixed":
                    if arch in ("gat", "transformer") and width % num_heads != 0:
                        raise ValueError(
                            f"For architecture '{arch}', width ({width}) must be "
                            f"divisible by num_heads ({num_heads}) in fixed mode."
                        )
                    chosen_width = width
                    model_cfg.message_hidden_dim = chosen_width
                    num_params = count_model_params(model_cfg)
                    print(
                        f"[ArchitectureComparison] Fixed width: {chosen_width}, "
                        f"params: {num_params}"
                    )
                else:  # normalized_params
                    chosen_width, reference_params = choose_architecture_width(
                        arch, depth, width, num_heads, base_model_config, num_targets,
                    )
                    model_cfg.message_hidden_dim = chosen_width
                    num_params = count_model_params(model_cfg)
                    print(
                        f"[ArchitectureComparison] Normalized width: {chosen_width} "
                        f"(reference GCN params: {reference_params}, "
                        f"this arch: {num_params})"
                    )

                config = Config(
                    data=data_cfg,
                    model=model_cfg,
                    training=copy.deepcopy(base_training_config),
                    experiment_name=(
                        f"arch_comp_{split_method}_{arch}_d{depth}_w{width}"
                    ),
                    description=(
                        f"Architecture comparison: {arch} on {split_method} split"
                    ),
                    tags=["architecture_comparison"],
                )
                try:
                    _, module, _, test_results_list = train_from_config(
                        config, num_workers=num_workers
                    )
                    test_results = (
                        test_results_list[0] if test_results_list else None
                    )
                    actual_num_params = sum(
                        p.numel() for p in module.model.parameters()
                    )
                except Exception as e:
                    print(
                        f"[ArchitectureComparison] Experiment failed for "
                        f"{arch} on {split_method}: {e}"
                    )
                    test_results = None
                    actual_num_params = 0
                row = build_arch_result_row(
                    split_method=split_method,
                    arch=arch,
                    depth=depth,
                    requested_width=width,
                    chosen_width=chosen_width,
                    num_params=actual_num_params,
                    test_results=test_results,
                )
                results.append(row)
                completed.add(key)
                df = pl.DataFrame(results)
                df.write_csv(ARCH_RESULTS_CSV)
                print(f"[ArchitectureComparison] Saved results to {ARCH_RESULTS_CSV}")

        if not results:
            print("No results were generated.")
            return pl.DataFrame()
        final_df = pl.DataFrame(results)
        for metric in ["test/mae", "test/rmse", "test/r2", "num_params"]:
            if metric not in final_df.columns:
                continue
            pivot = final_df.pivot(
                index="architecture", on="split_method", values=metric
            )
            for method in split_methods:
                if method not in pivot.columns:
                    pivot = pivot.with_columns(pl.lit(None).alias(method))
            pivot = pivot.select(["architecture"] + split_methods)
            print(f"\n--- Architecture comparison: {metric} ---")
            header = "| " + " | ".join(pivot.columns) + " |"
            separator = "|---" * len(pivot.columns) + "|"
            print(header)
            print(separator)
            for row_data in pivot.iter_rows():
                cells = []
                for val in row_data:
                    if val is None:
                        cells.append("NaN")
                    elif isinstance(val, float):
                        cells.append(f"{val:.4f}")
                    else:
                        cells.append(str(val))
                print("| " + " | ".join(cells) + " |")
        return final_df

    return (systematic_architecture_comparison,)


@app.cell
def _ui():
    import marimo as mo

    return (mo,)


@app.cell
def _size_controls(CSV_PATH, SIZE_RESULTS_CSV, mo):
    # Edit these Python variables to configure the size comparison.
    size_split = ["random"]
    size_arch = "gcn"
    size_depths = [4]
    size_widths = [128]
    size_epochs = 30
    size_batch = 128
    size_lr = 1e-3
    size_num_workers = 0
    RUN_SIZE = False

    csv_info = mo.md(f"**CSV path:** `{CSV_PATH}`")
    size_csv_info = mo.md(f"**Size results CSV:** `{SIZE_RESULTS_CSV}`")
    return (
        RUN_SIZE,
        csv_info,
        size_arch,
        size_batch,
        size_csv_info,
        size_depths,
        size_epochs,
        size_lr,
        size_num_workers,
        size_split,
        size_widths,
    )


@app.cell
def _arch_controls(ARCH_RESULTS_CSV, mo):
    # Edit these Python variables to configure the architecture comparison.
    arch_split = ["random","mces_umap"]
    arch_archs = {"gcn": True}
    arch_depth = 4
    arch_width = 256
    arch_epochs = 30
    arch_batch = 128
    arch_lr = 1e-3
    arch_num_workers = 0
    RUN_ARCH = False

    arch_csv_info = mo.md(f"**Arch results CSV:** `{ARCH_RESULTS_CSV}`")
    return (
        RUN_ARCH,
        arch_archs,
        arch_batch,
        arch_csv_info,
        arch_depth,
        arch_epochs,
        arch_lr,
        arch_num_workers,
        arch_split,
        arch_width,
    )


@app.cell
def _run_size(
    RUN_SIZE,
    size_arch,
    size_batch,
    size_depths,
    size_epochs,
    size_lr,
    size_num_workers,
    size_split,
    size_widths,
    systematic_size_comparison,
):
    size_df = None
    if RUN_SIZE:
        size_df = systematic_size_comparison(
            split_methods=list(size_split),
            depths=size_depths,
            widths=size_widths,
            num_epochs=int(size_epochs),
            batch_size=int(size_batch),
            learning_rate=float(size_lr),
            gnn_type=size_arch,
            num_workers=int(size_num_workers),
        )
    return (size_df,)


@app.cell
def _run_arch(
    RUN_ARCH,
    arch_archs,
    arch_batch,
    arch_depth,
    arch_epochs,
    arch_lr,
    arch_num_workers,
    arch_split,
    arch_width,
    systematic_architecture_comparison,
):
    arch_df = None
    if RUN_ARCH:
        arch_df = systematic_architecture_comparison(
            split_methods=list(arch_split),
            architectures=arch_archs,
            depth=int(arch_depth),
            width=int(arch_width),
            size_mode="normalized_params",
            num_epochs=int(arch_epochs),
            batch_size=int(arch_batch),
            learning_rate=float(arch_lr),
            num_workers=int(arch_num_workers),
        )
    return (arch_df,)


@app.cell
def _display_size(csv_info, mo, size_csv_info, size_df):
    if size_df is None:
        mo.vstack([
            csv_info,
            size_csv_info,
            mo.md(
                "Set `RUN_SIZE = True` to train models across split methods, "
                "depths, and widths. The notebook is resumable: it reads the "
                "results CSV and skips experiments that have already completed."
            ),
        ])
    else:
        size_df
    return


@app.cell
def _display_arch(arch_csv_info, arch_df, csv_info, mo):
    if arch_df is None:
        mo.vstack([
            csv_info,
            arch_csv_info,
            mo.md(
                "Set `RUN_ARCH = True` to train models across split methods "
                "and PyG architectures with `size_mode='normalized_params'`. "
                "The notebook is resumable."
            ),
        ])
    else:
        arch_df
    return


if __name__ == "__main__":
    app.run()
