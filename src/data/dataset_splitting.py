"""
Dataset splitting strategies for molecular data.

Implements various splitting methods including random, scaffold-based,
and Butina clustering-based splits.
"""

import polars as pl
import numpy as np
# import torch
from typing import Tuple, List, Optional
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from rdkit.Chem import rdFingerprintGenerator
from collections import defaultdict
import traceback
import jax
import jax.numpy as jnp
import time
import gc
from mces_splitting import split_dataset_lower_bound_only

def split_random(
    df: pl.DataFrame,
    test_fraction: float,
    val_fraction: float,
    seed: int
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
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


def split_scaffold(
    df: pl.DataFrame,
    test_fraction: float,
    val_fraction: float,
    seed: int,
    smiles_column: str = "smiles"
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Scaffold-based split using Bemis-Murcko scaffolds.
    
    Ensures molecules with the same scaffold are in the same split,
    which tests generalization to new scaffolds.
    
    Args:
        df: Input dataframe
        test_fraction: Fraction for test set
        val_fraction: Fraction for validation set
        seed: Random seed for reproducibility
        smiles_column: Name of column containing SMILES strings
    
    Returns:
        train_df, val_df, test_df
    """
    print("[split_scaffold] Computing Bemis-Murcko scaffolds...")
    
    # Compute scaffolds for all molecules
    scaffolds = []
    valid_indices = []
    
    for idx, row in enumerate(df.iter_rows(named=True)):
        smiles = row[smiles_column]
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            print(f"[Warning] Invalid SMILES at index {idx}, skipping")
            continue
        
        try:
            # Get Bemis-Murcko scaffold
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            scaffolds.append(scaffold)
            valid_indices.append(idx)
        except Exception as e:
            print(f"[Warning] Failed to compute scaffold at index {idx}: {e}")
            continue
    
    # Filter dataframe to valid molecules
    df_valid = df[valid_indices]
    
    # Group by scaffold
    scaffold_to_indices = defaultdict(list)
    for idx, scaffold in enumerate(scaffolds):
        scaffold_to_indices[scaffold].append(idx)
    
    # Sort scaffolds by size (largest first) for deterministic behavior
    scaffold_sets = sorted(scaffold_to_indices.values(), key=len, reverse=True)
    
    print(f"[split_scaffold] Found {len(scaffold_sets)} unique scaffolds")
    print(f"[split_scaffold] Largest scaffold has {len(scaffold_sets[0])} molecules")
    print(f"[split_scaffold] Smallest scaffold has {len(scaffold_sets[-1])} molecules")
    
    # Shuffle scaffold order with seed for reproducibility
    rng = np.random.RandomState(seed)
    rng.shuffle(scaffold_sets)
    
    # Allocate scaffolds to splits
    n = len(df_valid)
    test_size = int(n * test_fraction)
    val_size = int(n * val_fraction)
    
    train_indices = []
    val_indices = []
    test_indices = []
    
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
    
    # Create splits
    train_df = df_valid[train_indices]
    val_df = df_valid[val_indices]
    test_df = df_valid[test_indices]
    
    print(f"[split_scaffold] Final split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    return train_df, val_df, test_df


@jax.jit
def _compute_tanimoto_block_jax(
    fps_i: jnp.ndarray,
    fps_j: jnp.ndarray,
    counts_i: jnp.ndarray,
    counts_j: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute Tanimoto similarity for a block of fingerprints using JAX.
    
    Tanimoto similarity = |A ∩ B| / |A ∪ B|
                        = |A ∩ B| / (|A| + |B| - |A ∩ B|)
    
    Args:
        fps_i: Boolean array of shape (n_i, fingerprint_length)
        fps_j: Boolean array of shape (n_j, fingerprint_length)
        counts_i: Bit counts for fps_i, shape (n_i,)
        counts_j: Bit counts for fps_j, shape (n_j,)
        
    Returns:
        Similarity matrix of shape (n_i, n_j)
    """
    # Compute intersection using bitwise AND
    # Result shape: (n_i, n_j)
    intersection = jnp.sum(
        fps_i[:, None, :] & fps_j[None, :, :],
        axis=2,
        dtype=jnp.int32
    )
    
    # Compute union: |A| + |B| - |A ∩ B|
    union = counts_i[:, None] + counts_j[None, :] - intersection
    
    # Compute Tanimoto similarity, avoiding division by zero
    similarity = jnp.where(
        union > 0,
        intersection / union,
        0.0
    )
    
    return similarity


def _compute_tanimoto_distance_matrix_jax(
    fingerprints: List,
    batch_size: int = 1000
) -> np.ndarray:
    """
    Compute pairwise Tanimoto distance matrix using JAX GPU acceleration.
    Only computes upper triangle and stores results on CPU.
    """
    overall_start = time.time()
    n = len(fingerprints)
    
    # Convert fingerprints to JAX arrays (as before)
    conversion_start = time.time()
    fp_arrays = []
    for i, fp in enumerate(fingerprints):
        if i % 1000 == 0 and i > 0:
            print(f"Converting {i}/{n}...")
        arr = np.zeros((len(fp),), dtype=np.uint8)
        AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
        fp_arrays.append(arr.astype(bool))
    
    fp_np = np.stack(fp_arrays)
    fps_jax = jnp.array(fp_np, dtype=jnp.bool_)
    bit_counts = jnp.sum(fps_jax, axis=1, dtype=jnp.int32)
    jax.block_until_ready(bit_counts)
    
    print(f"[split_butina] Computing distance matrix on GPU (storing results on CPU)...")
    
    # Allocate full distance matrix on CPU (NumPy)
    dist_matrix_np = np.zeros((n, n), dtype=np.float32)
    
    computation_start = time.time()
    blocks_computed = 0
    total_blocks = sum(1 for i in range(0, n, batch_size) 
                      for j in range(i, n, batch_size))
    
    # Only compute upper triangle (i <= j)
    for i in range(0, n, batch_size):
        i_end = min(i + batch_size, n)
        batch_i = fps_jax[i:i_end]
        batch_counts_i = bit_counts[i:i_end]
        
        # Start from i (not 0) to only compute upper triangle
        for j in range(i, n, batch_size):
            j_end = min(j + batch_size, n)
            batch_j = fps_jax[j:j_end]
            batch_counts_j = bit_counts[j:j_end]
            
            # Compute similarity block on GPU
            sim_block = _compute_tanimoto_block_jax(
                batch_i, batch_j, batch_counts_i, batch_counts_j
            )
            dist_block = 1.0 - sim_block
            
            # Force computation and transfer to CPU
            dist_block_cpu = np.array(jax.block_until_ready(dist_block))
            
            # Store in upper triangle
            dist_matrix_np[i:i_end, j:j_end] = dist_block_cpu
            
            # Mirror to lower triangle (if not on diagonal)
            if i != j:
                dist_matrix_np[j:j_end, i:i_end] = dist_block_cpu.T
            
            blocks_computed += 1
            if blocks_computed % 10 == 0:
                progress = int(100 * blocks_computed / total_blocks)
                print(f"Progress: {progress}% ({blocks_computed}/{total_blocks})")
    
    print(f"[split_butina] Computation completed in {time.time() - computation_start:.2f}s")
    
    # Free GPU memory properly
    print(f"[split_butina] Freeing GPU resources...")
    del fps_jax, bit_counts, fp_np, fp_arrays
    
    # Clear JAX caches
    jax.clear_caches()
    
    # Force garbage collection
    gc.collect()
    
    # Wait for JAX to finish all operations
    jax.block_until_ready(jnp.array(0))
    
    print(f"[split_butina] GPU resources freed")
    
    return dist_matrix_np


def split_butina(
    df: pl.DataFrame,
    test_fraction: float,
    val_fraction: float,
    seed: int,
    smiles_column: str = "smiles",
    cutoff: float = 0.35,
    radius: int = 2,
    nbits: int = 2048,
    use_gpu: bool = True,
    batch_size: int = 1000
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Butina clustering-based split using Morgan fingerprints.
    
    Clusters molecules by structural similarity and ensures molecules
    in the same cluster are in the same split. Tests generalization
    to dissimilar molecules.
    
    Args:
        df: Input dataframe
        test_fraction: Fraction for test set
        val_fraction: Fraction for validation set
        seed: Random seed for reproducibility
        smiles_column: Name of column containing SMILES strings
        cutoff: Tanimoto distance threshold for clustering (default 0.35)
        radius: Morgan fingerprint radius (default 2)
        nbits: Morgan fingerprint size (default 2048)
        use_gpu: Whether to use GPU acceleration for distance matrix (default True)
        batch_size: Batch size for GPU computation (default 1000)
    
    Returns:
        train_df, val_df, test_df
    """
    print(f"[split_butina] Computing Morgan fingerprints (radius={radius}, nbits={nbits})...")
    
    # Create Morgan fingerprint generator
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    
    # Compute Morgan fingerprints
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
                print(f"[Warning] Failed to generate fingerprint at index {idx}, skipping")
                continue
            fingerprints.append(fp)
            valid_indices.append(idx)
        except Exception as e:
            print(f"[Warning] Failed to compute fingerprint at index {idx}: {e}")
            continue
    
    # Filter dataframe to valid molecules
    df_valid = df[valid_indices]
    n = len(df_valid)
    
    print(f"[split_butina] Successfully computed {n} fingerprints")
    print(f"[split_butina] Computing distance matrix for {n} molecules...")
    
    # Compute pairwise Tanimoto distances
    if use_gpu:
        print(f"[split_butina] Using JAX GPU-accelerated distance computation")
        try:
            distances = _compute_tanimoto_distance_matrix_jax(
                fingerprints, 
                batch_size=batch_size
            )
        except Exception as e:
            print(f"[Error] JAX GPU computation failed: {e}")
            traceback.print_exc()
            print(f"[split_butina] Falling back to CPU computation")
            use_gpu = False
    
    if not use_gpu:
        print(f"[split_butina] Using CPU distance computation")
        distances = []
        for i in range(n):
            if i % 100 == 0:
                print(f"[split_butina] Progress: {int(100 * i / n)}%")
            for j in range(i + 1, n):
                similarity = AllChem.DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                distance = 1.0 - similarity
                distances.append(distance)
    
    print(f"[split_butina] Clustering with cutoff={cutoff}...")
    
    # Perform Butina clustering
    clusters = Butina.ClusterData(distances, n, cutoff, isDistData=True)
    
    # Convert tuples to lists and sort by cluster size
    cluster_list = [list(cluster) for cluster in clusters]
    cluster_list.sort(key=len, reverse=True)
    
    print(f"[split_butina] Found {len(cluster_list)} clusters")
    print(f"[split_butina] Largest cluster: {len(cluster_list[0])} molecules")
    print(f"[split_butina] Smallest cluster: {len(cluster_list[-1])} molecules")
    
    # Shuffle cluster order with seed for reproducibility
    rng = np.random.RandomState(seed)
    rng.shuffle(cluster_list)
    
    # Allocate clusters to splits
    test_size = int(n * test_fraction)
    val_size = int(n * val_fraction)
    
    train_indices = []
    val_indices = []
    test_indices = []
    
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
    
    # Create splits
    train_df = df_valid[train_indices]
    val_df = df_valid[val_indices]
    test_df = df_valid[test_indices]
    
    print(f"[split_butina] Final split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    return train_df, val_df, test_df

def split_mces(
    df: pl.DataFrame,
    test_fraction: float,
    val_fraction: float,
    seed: int,
    smiles_column: str = "smiles",
    mces_matrix_save_path: Optional[str] = None
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split the dataset into train, validation, and test sets using the MCES lower bound method.
    """
    smiles_list = df[smiles_column].to_list()
    train_list, val_list, test_list, threshold = split_dataset_lower_bound_only(
        smiles_list,
        test_fraction=test_fraction,
        validation_fraction=val_fraction,
        initial_distinction_threshold=10,
        min_distinction_threshold=1,
        min_ratio=0.7,
        mces_matrix_save_path=mces_matrix_save_path
    )
    print(f"[split_mces] Using actual MCES threshold: {threshold}")
    train_df = df.filter(pl.col(smiles_column).is_in(train_list))
    val_df = df.filter(pl.col(smiles_column).is_in(val_list))
    test_df = df.filter(pl.col(smiles_column).is_in(test_list))
    return train_df, val_df, test_df