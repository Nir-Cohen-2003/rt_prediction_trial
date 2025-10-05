"""
Dataset splitting strategies for molecular data.

Implements various splitting methods including random, scaffold-based,
and Butina clustering-based splits.
"""

import polars as pl
import numpy as np
from typing import Tuple
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from collections import defaultdict


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
    inchi_column: str = "inchi"
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
        inchi_column: Name of column containing InChI strings
    
    Returns:
        train_df, val_df, test_df
    """
    print("[split_scaffold] Computing Bemis-Murcko scaffolds...")
    
    # Compute scaffolds for all molecules
    scaffolds = []
    valid_indices = []
    
    for idx, row in enumerate(df.iter_rows(named=True)):
        inchi = row[inchi_column]
        mol = Chem.MolFromInchi(inchi)
        
        if mol is None:
            print(f"[Warning] Invalid InChI at index {idx}, skipping")
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


def split_butina(
    df: pl.DataFrame,
    test_fraction: float,
    val_fraction: float,
    seed: int,
    inchi_column: str = "inchi",
    cutoff: float = 0.35,
    radius: int = 2,
    nbits: int = 2048
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
        inchi_column: Name of column containing InChI strings
        cutoff: Tanimoto distance threshold for clustering (default 0.35)
        radius: Morgan fingerprint radius (default 2)
        nbits: Morgan fingerprint size (default 2048)
    
    Returns:
        train_df, val_df, test_df
    """
    print(f"[split_butina] Computing Morgan fingerprints (radius={radius}, nbits={nbits})...")
    
    # Compute Morgan fingerprints
    fingerprints = []
    valid_indices = []
    
    for idx, row in enumerate(df.iter_rows(named=True)):
        inchi = row[inchi_column]
        mol = Chem.MolFromInchi(inchi)
        
        if mol is None:
            print(f"[Warning] Invalid InChI at index {idx}, skipping")
            continue
        
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
            fingerprints.append(fp)
            valid_indices.append(idx)
        except Exception as e:
            print(f"[Warning] Failed to compute fingerprint at index {idx}: {e}")
            continue
    
    # Filter dataframe to valid molecules
    df_valid = df[valid_indices]
    n = len(df_valid)
    
    print(f"[split_butina] Computing distance matrix for {n} molecules...")
    
    # Compute pairwise Tanimoto distances
    distances = []
    for i in range(n):
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


def split_custom(
    df: pl.DataFrame,
    splitter_fn,
    **kwargs
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
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