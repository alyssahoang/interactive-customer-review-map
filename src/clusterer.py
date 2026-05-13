"""Clustering wrappers and diagnostics for semantic embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


@dataclass
class ClusteringRun:
    """Container for one clustering execution and its quality metrics."""

    algorithm: str
    params: dict
    labels: np.ndarray
    metrics: dict


class ClusteringEngine:
    """Run K-Means, Spectral, and DBSCAN with unified metric reporting."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    @staticmethod
    def _cluster_count(labels: np.ndarray) -> int:
        unique = set(labels.tolist())
        if -1 in unique:
            unique.remove(-1)
        return len(unique)

    @staticmethod
    def _noise_ratio(labels: np.ndarray) -> float:
        return float(np.mean(labels == -1)) if len(labels) else 0.0

    @staticmethod
    def _evaluation_view(embeddings: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Internal metrics on non-noise points for density-based methods.
        if -1 in labels:
            mask = labels != -1
            if mask.sum() >= 2:
                return embeddings[mask], labels[mask]
        return embeddings, labels

    def internal_metrics(self, embeddings: np.ndarray, labels: np.ndarray) -> dict:
        eval_X, eval_y = self._evaluation_view(embeddings, labels)
        n_clusters = self._cluster_count(labels)

        metrics = {
            "n_clusters": n_clusters,
            "noise_ratio": self._noise_ratio(labels),
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
        }

        if len(eval_X) < 3 or len(set(eval_y.tolist())) < 2:
            return metrics

        metrics["silhouette"] = silhouette_score(eval_X, eval_y)
        metrics["calinski_harabasz"] = calinski_harabasz_score(eval_X, eval_y)
        metrics["davies_bouldin"] = davies_bouldin_score(eval_X, eval_y)
        return metrics

    def run_kmeans(self, embeddings: np.ndarray, n_clusters: int) -> ClusteringRun:
        model = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init="auto")
        labels = model.fit_predict(embeddings)
        metrics = self.internal_metrics(embeddings, labels)
        metrics["inertia"] = float(model.inertia_)
        return ClusteringRun(
            algorithm="kmeans",
            params={"n_clusters": n_clusters},
            labels=labels,
            metrics=metrics,
        )

    def run_dbscan(self, embeddings: np.ndarray, eps: float, min_samples: int) -> ClusteringRun:
        model = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
        labels = model.fit_predict(embeddings)
        metrics = self.internal_metrics(embeddings, labels)
        return ClusteringRun(
            algorithm="dbscan",
            params={"eps": eps, "min_samples": min_samples},
            labels=labels,
            metrics=metrics,
        )

    def run_spectral(self, embeddings: np.ndarray, n_clusters: int) -> ClusteringRun:
        model = SpectralClustering(
            n_clusters=n_clusters,
            random_state=self.random_state,
            affinity="nearest_neighbors",
            assign_labels="kmeans",
        )
        labels = model.fit_predict(embeddings)
        metrics = self.internal_metrics(embeddings, labels)
        return ClusteringRun(
            algorithm="spectral",
            params={"n_clusters": n_clusters},
            labels=labels,
            metrics=metrics,
        )

    def sweep_kmeans(self, embeddings: np.ndarray, k_values: Iterable[int]) -> pd.DataFrame:
        rows = []
        for k in k_values:
            run = self.run_kmeans(embeddings, n_clusters=int(k))
            row = {"algorithm": run.algorithm, **run.params, **run.metrics}
            rows.append(row)
        return pd.DataFrame(rows).sort_values("n_clusters").reset_index(drop=True)

    def sweep_dbscan(
        self,
        embeddings: np.ndarray,
        eps_values: Iterable[float],
        min_samples_values: Iterable[int],
    ) -> pd.DataFrame:
        rows = []
        for eps in eps_values:
            for min_samples in min_samples_values:
                run = self.run_dbscan(embeddings, eps=float(eps), min_samples=int(min_samples))
                row = {"algorithm": run.algorithm, **run.params, **run.metrics}
                rows.append(row)
        return pd.DataFrame(rows).sort_values(["eps", "min_samples"]).reset_index(drop=True)

    @staticmethod
    def select_best(
        score_df: pd.DataFrame,
        metric: str = "silhouette",
        maximize: bool = True,
    ) -> pd.Series:
        valid = score_df.dropna(subset=[metric]).copy()
        if valid.empty:
            raise ValueError(f"No valid rows for metric '{metric}'.")
        idx = valid[metric].idxmax() if maximize else valid[metric].idxmin()
        return valid.loc[idx]
