"""Evaluation helpers for sentiment/topic behavior in semantic geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class GeometryEvaluator:
    """Compute cluster-level and pairwise diagnostics for mapped reviews."""

    random_state: int = 42

    @staticmethod
    def _cluster_ids(labels: np.ndarray) -> list[int]:
        return sorted(int(x) for x in np.unique(labels) if x != -1)

    def sentiment_by_cluster(self, labels: np.ndarray, sentiments: pd.Series) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "cluster_id": labels,
                "sentiment": sentiments.reset_index(drop=True),
            }
        )
        table = (
            frame.groupby(["cluster_id", "sentiment"])
            .size()
            .rename("count")
            .reset_index()
            .pivot(index="cluster_id", columns="sentiment", values="count")
            .fillna(0)
        )
        table = table.div(table.sum(axis=1), axis=0).fillna(0)
        return table.reset_index()

    def cluster_dispersion(self, embeddings: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
        rows = []
        labels = np.asarray(labels)

        for cluster_id in self._cluster_ids(labels):
            idx = np.where(labels == cluster_id)[0]
            cluster_vectors = embeddings[idx]
            centroid = cluster_vectors.mean(axis=0, keepdims=True)
            sims = cosine_similarity(cluster_vectors, centroid).ravel()

            row = {
                "cluster_id": cluster_id,
                "size": int(len(idx)),
                "mean_cosine_similarity_to_centroid": float(np.mean(sims)),
                "dispersion": float(1 - np.mean(sims)),
            }
            rows.append(row)

        return pd.DataFrame(rows).sort_values("dispersion", ascending=False).reset_index(drop=True)

    def negative_profile(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        sentiments: pd.Series,
    ) -> pd.DataFrame:
        labels = np.asarray(labels)
        sentiments = sentiments.reset_index(drop=True)

        dispersion_df = self.cluster_dispersion(embeddings, labels)
        ratios = []
        for cluster_id in dispersion_df["cluster_id"]:
            idx = np.where(labels == cluster_id)[0]
            ratio = float((sentiments.iloc[idx] == "negative").mean())
            ratios.append(ratio)
        dispersion_df["negative_ratio"] = ratios
        return dispersion_df

    def bootstrap_negative_ratio_ci(
        self,
        labels: np.ndarray,
        sentiments: pd.Series,
        cluster_id: int,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
    ) -> dict:
        labels = np.asarray(labels)
        sentiments = sentiments.reset_index(drop=True).to_numpy()

        idx = np.where(labels == cluster_id)[0]
        if len(idx) == 0:
            return {
                "cluster_id": cluster_id,
                "estimate": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "n": 0,
            }

        rng = np.random.default_rng(self.random_state)
        observed = float(np.mean(sentiments[idx] == "negative"))
        samples = []

        for _ in range(n_bootstrap):
            boot_idx = rng.choice(idx, size=len(idx), replace=True)
            samples.append(float(np.mean(sentiments[boot_idx] == "negative")))

        alpha = 1.0 - confidence
        ci_low = float(np.quantile(samples, alpha / 2))
        ci_high = float(np.quantile(samples, 1 - alpha / 2))

        return {
            "cluster_id": cluster_id,
            "estimate": observed,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n": int(len(idx)),
        }

    def bootstrap_profile(
        self,
        labels: np.ndarray,
        sentiments: pd.Series,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
    ) -> pd.DataFrame:
        rows = []
        for cluster_id in self._cluster_ids(np.asarray(labels)):
            rows.append(
                self.bootstrap_negative_ratio_ci(
                    labels=labels,
                    sentiments=sentiments,
                    cluster_id=cluster_id,
                    n_bootstrap=n_bootstrap,
                    confidence=confidence,
                )
            )
        return pd.DataFrame(rows).sort_values("estimate", ascending=False).reset_index(drop=True)
