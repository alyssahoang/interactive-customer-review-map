"""Cluster-level topic extraction utilities based on keyword and exemplar signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ClusterTopicModeler:
    """Build lightweight topic labels and example snippets from clustered reviews."""

    n_keywords: int = 5
    max_exemplars: int = 5
    random_state: int = 42

    @staticmethod
    def _cluster_ids(labels: np.ndarray) -> list[int]:
        cluster_ids = sorted(int(x) for x in np.unique(labels) if x != -1)
        return cluster_ids

    def select_exemplars(
        self,
        texts: pd.Series,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[int, list[str]]:
        exemplars: Dict[int, list[str]] = {}
        text_values = texts.reset_index(drop=True)
        labels = np.asarray(labels)

        for cluster_id in self._cluster_ids(labels):
            idx = np.where(labels == cluster_id)[0]
            cluster_vectors = embeddings[idx]
            centroid = cluster_vectors.mean(axis=0, keepdims=True)
            sims = cosine_similarity(cluster_vectors, centroid).ravel()
            top_local = np.argsort(sims)[::-1][: self.max_exemplars]
            top_global = idx[top_local]
            exemplars[cluster_id] = text_values.iloc[top_global].tolist()
        return exemplars

    def keyword_topic_names(self, texts: pd.Series, labels: np.ndarray) -> Dict[int, str]:
        text_values = texts.fillna("").astype(str).reset_index(drop=True)
        labels = np.asarray(labels)

        try:
            vectorizer = TfidfVectorizer(max_features=12000, ngram_range=(1, 2), min_df=2)
            tfidf = vectorizer.fit_transform(text_values)
        except ValueError:
            vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=1)
            tfidf = vectorizer.fit_transform(text_values)

        terms = np.array(vectorizer.get_feature_names_out())
        names: Dict[int, str] = {}

        for cluster_id in self._cluster_ids(labels):
            idx = np.where(labels == cluster_id)[0]
            cluster_mean = np.asarray(tfidf[idx].mean(axis=0)).ravel()
            top_idx = cluster_mean.argsort()[::-1][: self.n_keywords]
            top_terms = [term for term, score in zip(terms[top_idx], cluster_mean[top_idx]) if score > 0]
            names[cluster_id] = " | ".join(top_terms) if top_terms else f"cluster_{cluster_id}"

        return names

    def build_topic_table(
        self,
        texts: pd.Series,
        embeddings: np.ndarray,
        labels: np.ndarray,
        sentiment_labels: pd.Series | None = None,
    ) -> pd.DataFrame:
        labels = np.asarray(labels)
        text_values = texts.fillna("").astype(str).reset_index(drop=True)
        topic_names = self.keyword_topic_names(text_values, labels)
        exemplars = self.select_exemplars(text_values, embeddings, labels)

        rows = []
        for cluster_id in self._cluster_ids(labels):
            idx = np.where(labels == cluster_id)[0]
            row = {
                "cluster_id": cluster_id,
                "size": int(len(idx)),
                "topic_name": topic_names.get(cluster_id, f"cluster_{cluster_id}"),
                "example_reviews": exemplars.get(cluster_id, []),
            }
            if sentiment_labels is not None:
                cluster_sentiments = sentiment_labels.reset_index(drop=True).iloc[idx]
                row["negative_ratio"] = float((cluster_sentiments == "negative").mean())
            rows.append(row)

        return pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)
