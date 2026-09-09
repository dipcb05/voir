"""
Data partitioning for federated learning (IID and non-IID).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def create_federated_partitions(
    df: pd.DataFrame,
    num_clients: int,
    partition_type: str = "iid",
    label_col: str = "label",
    alpha: float = 0.5,
    seed: int = 42,
) -> Dict[int, pd.DataFrame]:
    """
    Partition dataframe into FL clients.

    Args:
        df: Input dataframe.
        num_clients: Number of clients.
        partition_type: 'iid' or 'non_iid_dirichlet'.
        label_col: Target label column name.
        alpha: Dirichlet concentration parameter for non-IID.
        seed: Random seed.

    Returns:
        Dict mapping client_id (0 to num_clients-1) to partitioned DataFrame.
    """
    rng = np.random.RandomState(seed)
    partitions: Dict[int, pd.DataFrame] = {}

    if partition_type.lower() == "iid":
        # Random shuffle and split evenly
        shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        splits = np.array_split(shuffled, num_clients)
        for i in range(num_clients):
            partitions[i] = splits[i].reset_index(drop=True)

    elif partition_type.lower() == "non_iid_dirichlet":
        # Dirichlet distribution over labels
        labels = df[label_col].values
        num_classes = len(np.unique(labels))
        
        client_indices: List[List[int]] = [[] for _ in range(num_clients)]
        
        for k in range(num_classes):
            idx_k = np.where(labels == k)[0]
            rng.shuffle(idx_k)
            
            # Sample proportions for this class across clients
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            
            # Balance out to avoid 0 samples in some clients for a class
            proportions = np.array([p * (len(idx_j) < len(labels) / num_clients) 
                                  for p, idx_j in zip(proportions, client_indices)])
            proportions = proportions / proportions.sum()
            
            # Split indices based on proportions
            splits = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_k_split = np.split(idx_k, splits)
            
            for i in range(num_clients):
                client_indices[i].extend(idx_k_split[i])
                
        for i in range(num_clients):
            rng.shuffle(client_indices[i])
            partitions[i] = df.iloc[client_indices[i]].reset_index(drop=True)
            
    else:
        raise ValueError(f"Unknown partition_type: {partition_type}")

    return partitions
