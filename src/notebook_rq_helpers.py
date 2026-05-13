"""RQ diagnostics helpers extracted from the notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def dispersion_profile(emb_matrix: np.ndarray, target_labels: np.ndarray) -> pd.DataFrame:
    """Group-wise mean pairwise cosine distance."""
    rows = []
    labels_arr = pd.Series(target_labels).fillna("unknown").astype(str).to_numpy()
    for group in pd.unique(labels_arr):
        idx = np.where(labels_arr == group)[0]
        n = len(idx)
        if n <= 1:
            continue
        group_vectors = emb_matrix[idx]
        sum_vec = group_vectors.sum(axis=0)
        sum_sq_norm = float(np.sum(group_vectors * group_vectors))
        pair_sum = float((sum_vec @ sum_vec - sum_sq_norm) / 2.0)
        pair_count = n * (n - 1) / 2.0
        mean_sim = pair_sum / pair_count
        rows.append(
            {
                "group": group,
                "mean_pairwise_cosine_distance": float(1.0 - mean_sim),
                "n_samples": int(n),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_pairwise_cosine_distance", ascending=False).reset_index(drop=True)


def negative_homophily(emb_matrix: np.ndarray, sentiments: np.ndarray, k: int = 20) -> tuple[float, float, float]:
    """Nearest-neighbor homophily for negative sentiment."""
    y = (pd.Series(sentiments).astype(str).to_numpy() == "negative").astype(np.int8)
    neg_idx = np.where(y == 1)[0]
    if len(neg_idx) == 0:
        return np.nan, float(y.mean()), np.nan

    n_neighbors = min(k + 1, len(emb_matrix))
    if n_neighbors <= 1:
        return np.nan, float(y.mean()), np.nan

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(emb_matrix)
    _, neigh_idx = nn.kneighbors(emb_matrix[neg_idx])
    neigh_idx = neigh_idx[:, 1:]
    if neigh_idx.size == 0:
        return np.nan, float(y.mean()), np.nan
    homophily_share = float(y[neigh_idx].mean())
    base_negative = float(y.mean())
    return homophily_share, base_negative, float(homophily_share - base_negative)


def bootstrap_cluster_neg_std(labels: np.ndarray, neg_binary: np.ndarray, boot_indices: list[np.ndarray]) -> np.ndarray:
    """Bootstrap dispersion of cluster-level negative ratios."""
    out = []
    for idx in boot_indices:
        frame = pd.DataFrame({"cluster": labels[idx], "neg": neg_binary[idx]})
        out.append(float(frame.groupby("cluster")["neg"].mean().std()))
    return np.asarray(out, dtype=float)


def safe_float(series_like, key, default=np.nan) -> float:
    """Safe extraction of float values from dict/Series objects."""
    try:
        v = series_like[key]
        return float(v)
    except Exception:
        return float(default)


def safe_int(series_like, key, default=0) -> int:
    """Safe extraction of integer values from dict/Series objects."""
    try:
        v = series_like[key]
        return int(v)
    except Exception:
        return int(default)

