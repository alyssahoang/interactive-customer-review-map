"""Cluster stability analysis routines for K-Means across random seeds."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score


@dataclass
class ClusterStabilityAnalyzer:
    """Robust K selection from repeated random seeds."""

    n_init: int = 10

    def evaluate_kmeans(
        self,
        embeddings: np.ndarray,
        k_values: Iterable[int],
        seeds: Iterable[int],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        rows = []
        label_store: dict[int, dict[int, np.ndarray]] = {}

        for k in k_values:
            k = int(k)
            label_store[k] = {}
            for seed in seeds:
                seed = int(seed)
                model = KMeans(n_clusters=k, random_state=seed, n_init=self.n_init)
                labels = model.fit_predict(embeddings)
                sil = silhouette_score(embeddings, labels) if k > 1 else np.nan

                rows.append(
                    {
                        "k": k,
                        "seed": seed,
                        "silhouette": float(sil),
                        "inertia": float(model.inertia_),
                    }
                )
                label_store[k][seed] = labels

        run_df = pd.DataFrame(rows).sort_values(["k", "seed"]).reset_index(drop=True)
        summary_df = (
            run_df.groupby("k")
            .agg(
                silhouette_mean=("silhouette", "mean"),
                silhouette_std=("silhouette", "std"),
                inertia_mean=("inertia", "mean"),
                inertia_std=("inertia", "std"),
                n_runs=("seed", "count"),
            )
            .reset_index()
        )

        ari_rows = []
        for k, per_seed in label_store.items():
            scores = []
            for s1, s2 in combinations(per_seed.keys(), 2):
                scores.append(adjusted_rand_score(per_seed[s1], per_seed[s2]))
            ari_rows.append(
                {
                    "k": k,
                    "ari_mean": float(np.mean(scores)) if scores else np.nan,
                    "ari_std": float(np.std(scores)) if scores else np.nan,
                }
            )
        ari_df = pd.DataFrame(ari_rows)

        summary_df = summary_df.merge(ari_df, on="k", how="left")
        summary_df["robust_score"] = (
            summary_df["silhouette_mean"]
            - summary_df["silhouette_std"].fillna(0)
            + 0.10 * summary_df["ari_mean"].fillna(0)
        )
        summary_df = summary_df.sort_values("robust_score", ascending=False).reset_index(drop=True)
        return run_df, summary_df

    @staticmethod
    def select_best_k(summary_df: pd.DataFrame) -> int:
        if summary_df.empty:
            raise ValueError("Stability summary is empty.")
        return int(summary_df.iloc[0]["k"])
