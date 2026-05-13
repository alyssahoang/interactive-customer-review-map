"""Interpretability probes for prototypes, nearest neighbors, and counterfactuals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


@dataclass
class SemanticInspector:
    """Prototype and counterfactual inspection for embedding spaces."""

    def cluster_prototypes(
        self,
        texts: pd.Series,
        embeddings: np.ndarray,
        labels: np.ndarray,
        n_per_cluster: int = 3,
    ) -> pd.DataFrame:
        labels = np.asarray(labels)
        text_values = texts.fillna("").astype(str).reset_index(drop=True)
        rows = []

        for cluster_id in sorted(int(x) for x in np.unique(labels) if x != -1):
            idx = np.where(labels == cluster_id)[0]
            vectors = embeddings[idx]
            centroid = vectors.mean(axis=0, keepdims=True)
            sims = cosine_similarity(vectors, centroid).ravel()
            top_local_idx = np.argsort(sims)[::-1][:n_per_cluster]

            for rank, local_idx in enumerate(top_local_idx, start=1):
                global_idx = idx[local_idx]
                rows.append(
                    {
                        "cluster_id": cluster_id,
                        "rank": rank,
                        "cosine_to_centroid": float(sims[local_idx]),
                        "text": text_values.iloc[global_idx],
                    }
                )

        return pd.DataFrame(rows).sort_values(["cluster_id", "rank"]).reset_index(drop=True)

    def nearest_counterfactuals(
        self,
        texts: pd.Series,
        embeddings: np.ndarray,
        sentiments: pd.Series,
        labels: np.ndarray | None = None,
        n_neighbors: int = 25,
        max_rows: int = 20,
    ) -> pd.DataFrame:
        text_values = texts.fillna("").astype(str).reset_index(drop=True)
        sent_values = sentiments.fillna("unknown").astype(str).reset_index(drop=True)
        label_values = (
            np.full(len(text_values), -1, dtype=int)
            if labels is None
            else np.asarray(labels).astype(int)
        )

        if len(text_values) < 2:
            return pd.DataFrame(
                columns=[
                    "source_idx",
                    "counterfactual_idx",
                    "source_sentiment",
                    "counterfactual_sentiment",
                    "source_cluster",
                    "counterfactual_cluster",
                    "cosine_similarity",
                    "source_text",
                    "counterfactual_text",
                ]
            )

        n_neighbors = min(max(2, n_neighbors), len(text_values))
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        knn.fit(embeddings)
        distances, indices = knn.kneighbors(embeddings)

        rows = []
        for i in range(len(text_values)):
            source_sent = sent_values.iloc[i]
            source_cluster = int(label_values[i])

            candidate = None
            for dist, j in zip(distances[i][1:], indices[i][1:]):
                if sent_values.iloc[j] != source_sent:
                    candidate = (float(1.0 - dist), int(j))
                    break

            if candidate is None:
                continue

            sim, j = candidate
            rows.append(
                {
                    "source_idx": int(i),
                    "counterfactual_idx": int(j),
                    "source_sentiment": source_sent,
                    "counterfactual_sentiment": sent_values.iloc[j],
                    "source_cluster": source_cluster,
                    "counterfactual_cluster": int(label_values[j]),
                    "cosine_similarity": sim,
                    "source_text": text_values.iloc[i],
                    "counterfactual_text": text_values.iloc[j],
                }
            )

        if not rows:
            return pd.DataFrame()

        pairs = pd.DataFrame(rows).sort_values("cosine_similarity", ascending=False)
        return pairs.head(max_rows).reset_index(drop=True)
