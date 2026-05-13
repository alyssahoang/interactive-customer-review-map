"""Notebook helpers for embedding extraction and Phase 3 diagnostics."""

from __future__ import annotations

from pathlib import Path
import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import normalize

from .experiment_config import ExperimentConfig


class SemanticEmbedder:
    """Extract dense semantic vectors with SentenceTransformer backends."""

    MODEL_MAP = {
        "bertimbau": "neuralmind/bert-base-portuguese-cased",
        "mbert": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "clip": "sentence-transformers/clip-ViT-B-32-multilingual-v1",
    }

    def __init__(self, model_type: str = "bertimbau", batch_size: int = 32, max_seq_length: int = 128):
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_type = model_type.lower()
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.model_type not in self.MODEL_MAP:
            raise ValueError(f"model_type must be one of {list(self.MODEL_MAP.keys())}")

        model_name = self.MODEL_MAP[self.model_type]
        logging.info("Loading model: %s on %s", model_name, self.device)
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.max_seq_length = self.max_seq_length

    def get_embeddings(self, texts: list[str], apply_l2_norm: bool = False) -> np.ndarray:
        """Encode text list into dense vectors."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        logging.info("Encoding %s texts with batch_size=%s", f"{len(texts):,}", self.batch_size)
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        if apply_l2_norm:
            vectors = normalize(vectors, norm="l2")
        return vectors.astype(np.float32)


def load_cached_l2(emb_dir: Path, model_key: str, n_rows_hint: int | None = None):
    """Load latest cached L2 embedding file for a model."""
    candidates = []
    if n_rows_hint is not None:
        candidates.extend(emb_dir.glob(f"{model_key}_n{n_rows_hint}_d*_l2.npy"))
    candidates.extend(emb_dir.glob(f"{model_key}_n*_d*_l2.npy"))
    if not candidates:
        return None, None
    chosen = sorted(set(candidates), key=lambda x: x.stat().st_mtime, reverse=True)[0]
    return np.load(chosen).astype(np.float32), chosen


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    """Read CSV or return empty DataFrame if missing/empty."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


class GeometryClusteringEvaluator:
    """Unsupervised clustering diagnostics for semantic geometry."""

    def __init__(self, random_state: int = 42, silhouette_sample_size: int | None = None):
        self.random_state = random_state
        self.silhouette_sample_size = silhouette_sample_size

    def _resolve_sample_size(self, n_rows: int, sample_size: int | None = None) -> int | None:
        ss = self.silhouette_sample_size if sample_size is None else sample_size
        if ss is None:
            return None
        ss = int(ss)
        if ss <= 1 or ss >= int(n_rows):
            return None
        return ss

    def _silhouette_cosine(self, X: np.ndarray, labels: np.ndarray, sample_size: int | None = None) -> float:
        labels_arr = np.asarray(labels)
        if len(labels_arr) != len(X):
            raise ValueError("labels length must match X length")
        if len(X) < 3 or np.unique(labels_arr).size < 2:
            return float("nan")
        ss = self._resolve_sample_size(len(X), sample_size=sample_size)
        try:
            return float(
                silhouette_score(
                    X,
                    labels_arr,
                    metric="cosine",
                    sample_size=ss,
                    random_state=self.random_state,
                )
            )
        except ValueError:
            return float("nan")

    def run_k_curve_diagnostics(
        self,
        X: np.ndarray,
        k_min: int = 2,
        k_max: int = 10,
        silhouette_sample_size: int | None = None,
    ) -> pd.DataFrame:
        rows: list[dict] = []
        for k in range(k_min, k_max + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            rows.append(
                {
                    "k": int(k),
                    "silhouette_cosine": self._silhouette_cosine(X, labels, sample_size=silhouette_sample_size),
                    "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
                    "davies_bouldin": float(davies_bouldin_score(X, labels)),
                    "inertia": float(kmeans.inertia_),
                }
            )
        return pd.DataFrame(rows)

    def run_spectral(
        self,
        X: np.ndarray,
        n_clusters: int,
        n_neighbors: int = 15,
        silhouette_sample_size: int | None = None,
    ):
        n_neighbors = max(2, min(n_neighbors, len(X) - 1))
        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=n_neighbors,
            assign_labels="kmeans",
            random_state=self.random_state,
        )
        labels = model.fit_predict(X)
        metrics = {
            "algorithm": "spectral",
            "k": int(n_clusters),
            "silhouette_cosine": self._silhouette_cosine(X, labels, sample_size=silhouette_sample_size),
            "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
            "davies_bouldin": float(davies_bouldin_score(X, labels)),
        }
        return labels, metrics

    def calculate_cluster_purity(self, cluster_labels: np.ndarray, target_labels: np.ndarray) -> float:
        frame = pd.DataFrame({"cluster": cluster_labels, "target": target_labels})
        frame = frame[frame["cluster"] != -1]
        if frame.empty:
            return np.nan
        dominant_counts = frame.groupby("cluster")["target"].value_counts().groupby(level=0).max()
        return float(dominant_counts.sum() / len(frame))


def stratified_sentiment_indices(
    sentiment_labels: np.ndarray,
    n_rows: int,
    sample_n: int | None,
    seed: int,
) -> np.ndarray:
    """Sample indices stratified by sentiment labels when available."""
    if sample_n is None or sample_n >= n_rows:
        return np.arange(n_rows, dtype=int)

    rng = np.random.default_rng(seed)
    sample_n = max(2, int(sample_n))

    if len(sentiment_labels) != n_rows:
        return np.sort(rng.choice(n_rows, size=sample_n, replace=False))

    y = pd.Series(sentiment_labels[:n_rows]).astype(str).reset_index(drop=True)
    picked_parts = []
    for _, idx in y.groupby(y).groups.items():
        cls_idx = np.asarray(list(idx), dtype=int)
        take = max(1, int(round(sample_n * len(cls_idx) / n_rows)))
        take = min(take, len(cls_idx))
        picked_parts.append(rng.choice(cls_idx, size=take, replace=False))

    picked = np.unique(np.concatenate(picked_parts)) if picked_parts else np.array([], dtype=int)
    if len(picked) > sample_n:
        picked = np.sort(rng.choice(picked, size=sample_n, replace=False))
    elif len(picked) < sample_n:
        remain = np.setdiff1d(np.arange(n_rows, dtype=int), picked, assume_unique=False)
        need = min(sample_n - len(picked), len(remain))
        if need > 0:
            extra = rng.choice(remain, size=need, replace=False)
            picked = np.sort(np.concatenate([picked, extra]))
    return picked


def run_k_grid_diagnostics(
    X_diag: np.ndarray,
    k_values: tuple[int, ...],
    random_state: int,
    geom_eval: GeometryClusteringEvaluator,
    silhouette_sample_size: int | None,
) -> pd.DataFrame:
    """Run KMeans metrics across an explicit k-grid."""
    rows = []
    for k in sorted(set(int(v) for v in k_values)):
        if k < 2 or k >= len(X_diag):
            continue
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_diag)
        rows.append(
            {
                "k": int(k),
                "silhouette_cosine": geom_eval._silhouette_cosine(
                    X_diag,
                    labels,
                    sample_size=silhouette_sample_size,
                ),
                "calinski_harabasz": float(calinski_harabasz_score(X_diag, labels)),
                "davies_bouldin": float(davies_bouldin_score(X_diag, labels)),
                "inertia": float(km.inertia_),
            }
        )
    if not rows:
        raise ValueError(f"No valid k values for diagnostics. Grid={k_values}, n_rows_diag={len(X_diag)}")
    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def select_best_k_ranked(k_scores: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    """Pick k by rank aggregation of silhouette/CH/DB metrics."""
    if k_scores.empty:
        raise ValueError("K-diagnostic table is empty; cannot select best k.")
    kdf = k_scores.copy().sort_values("k").reset_index(drop=True)
    kdf["rank_sil"] = kdf["silhouette_cosine"].rank(ascending=False, method="min")
    kdf["rank_ch"] = kdf["calinski_harabasz"].rank(ascending=False, method="min")
    kdf["rank_db"] = kdf["davies_bouldin"].rank(ascending=True, method="min")
    kdf["rank_sum"] = kdf[["rank_sil", "rank_ch", "rank_db"]].sum(axis=1)
    best_row = kdf.sort_values(["rank_sum", "k"]).iloc[0]
    return int(best_row["k"]), kdf


def resolve_k_for_model(
    model_key: str,
    k_scores_ranked: pd.DataFrame,
    k_auto: int,
    k_mode: str,
    k_fixed: int,
    k_by_model: dict[str, int],
) -> tuple[int, str]:
    """Resolve final k by mode with guardrails to evaluated candidates."""
    k_candidates = sorted(k_scores_ranked["k"].dropna().astype(int).unique().tolist())
    if not k_candidates:
        raise ValueError(f"No valid k candidates for model '{model_key}'.")

    if k_mode == "auto_ranked":
        return int(k_auto), "auto_ranked"
    if k_mode == "fixed":
        chosen = int(k_fixed)
        mode = "fixed"
    elif k_mode == "per_model":
        if model_key not in k_by_model:
            raise KeyError(f"model_key='{model_key}' missing in k_by_model")
        chosen = int(k_by_model[model_key])
        mode = "per_model"
    else:
        raise ValueError(f"Unsupported k_mode={k_mode!r}. Use auto_ranked, fixed, or per_model.")

    if chosen not in k_candidates:
        nearest = min(k_candidates, key=lambda k: abs(k - chosen))
        print(f"[{model_key}] requested k={chosen} not in evaluated grid {k_candidates}; using nearest k={nearest}.")
        chosen = int(nearest)
    return int(chosen), mode


def run_phase3_for_model(
    model_key: str,
    embeddings_l2_by_model: dict[str, np.ndarray],
    sentiment_labels: np.ndarray,
    topic_proxy: np.ndarray,
    cfg: ExperimentConfig,
    geom_eval: GeometryClusteringEvaluator,
    k_grid: tuple[int, ...],
    k_diag_sample_n: int,
    k_mode: str,
    k_fixed: int,
    k_by_model: dict[str, int],
    silhouette_sample_size: int | None,
) -> dict:
    """Execute Phase 3 clustering diagnostics for one model."""
    X_model = embeddings_l2_by_model[model_key]
    diag_idx = stratified_sentiment_indices(
        sentiment_labels=sentiment_labels,
        n_rows=len(X_model),
        sample_n=k_diag_sample_n,
        seed=cfg.random_state,
    )
    X_diag = X_model[diag_idx]

    kmeans_scores_model = run_k_grid_diagnostics(
        X_diag=X_diag,
        k_values=k_grid,
        random_state=cfg.random_state,
        geom_eval=geom_eval,
        silhouette_sample_size=silhouette_sample_size,
    )
    k_auto_model, kmeans_scores_ranked = select_best_k_ranked(kmeans_scores_model)
    best_k_model, k_mode_used = resolve_k_for_model(
        model_key=model_key,
        k_scores_ranked=kmeans_scores_ranked,
        k_auto=k_auto_model,
        k_mode=k_mode,
        k_fixed=k_fixed,
        k_by_model=k_by_model,
    )
    if int(best_k_model) != int(k_auto_model):
        print(f"[{model_key}] auto-ranked k={k_auto_model}; using {k_mode_used} k={best_k_model}.")

    km = KMeans(n_clusters=best_k_model, random_state=cfg.random_state, n_init=10)
    labels_model = km.fit_predict(X_model)
    kmeans_final_metrics_model = {
        "model_key": model_key,
        "algorithm": "kmeans",
        "k": int(best_k_model),
        "silhouette_cosine": geom_eval._silhouette_cosine(
            X_model,
            labels_model,
            sample_size=silhouette_sample_size,
        ),
        "calinski_harabasz": float(calinski_harabasz_score(X_model, labels_model)),
        "davies_bouldin": float(davies_bouldin_score(X_model, labels_model)),
        "inertia": float(km.inertia_),
    }

    spectral_labels_model, spectral_metrics_model = geom_eval.run_spectral(
        X_model,
        n_clusters=best_k_model,
        n_neighbors=15,
        silhouette_sample_size=silhouette_sample_size,
    )
    spectral_metrics_model["model_key"] = model_key

    purity_table_model = pd.DataFrame(
        [
            {
                "model_key": model_key,
                "algorithm": "kmeans",
                "purity_sentiment": geom_eval.calculate_cluster_purity(labels_model, sentiment_labels),
                "purity_topic_proxy": geom_eval.calculate_cluster_purity(labels_model, topic_proxy),
            },
            {
                "model_key": model_key,
                "algorithm": "spectral",
                "purity_sentiment": geom_eval.calculate_cluster_purity(spectral_labels_model, sentiment_labels),
                "purity_topic_proxy": geom_eval.calculate_cluster_purity(spectral_labels_model, topic_proxy),
            },
        ]
    )

    stability_rows = []
    for seed in cfg.stability_seeds:
        km_seed = KMeans(n_clusters=best_k_model, random_state=int(seed), n_init=10)
        seed_labels = km_seed.fit_predict(X_model)
        stability_rows.append(
            {
                "model_key": model_key,
                "seed": int(seed),
                "k": int(best_k_model),
                "silhouette_cosine": geom_eval._silhouette_cosine(
                    X_model,
                    seed_labels,
                    sample_size=silhouette_sample_size,
                ),
                "calinski_harabasz": float(calinski_harabasz_score(X_model, seed_labels)),
                "davies_bouldin": float(davies_bouldin_score(X_model, seed_labels)),
                "inertia": float(km_seed.inertia_),
            }
        )
    stability_runs_model = pd.DataFrame(stability_rows).sort_values("seed").reset_index(drop=True)
    stability_summary_model = pd.DataFrame(
        [
            {
                "model_key": model_key,
                "k": int(best_k_model),
                "seeds_tested": int(len(stability_runs_model)),
                "silhouette_mean": float(stability_runs_model["silhouette_cosine"].mean()),
                "silhouette_std": float(stability_runs_model["silhouette_cosine"].std(ddof=0)),
                "calinski_mean": float(stability_runs_model["calinski_harabasz"].mean()),
                "calinski_std": float(stability_runs_model["calinski_harabasz"].std(ddof=0)),
                "davies_mean": float(stability_runs_model["davies_bouldin"].mean()),
                "davies_std": float(stability_runs_model["davies_bouldin"].std(ddof=0)),
                "inertia_mean": float(stability_runs_model["inertia"].mean()),
                "inertia_std": float(stability_runs_model["inertia"].std(ddof=0)),
            }
        ]
    )

    cluster_size_df_model = (
        pd.Series(labels_model).value_counts().sort_index().rename_axis("cluster_id").reset_index(name="count")
    )
    cluster_size_df_model["ratio"] = cluster_size_df_model["count"] / cluster_size_df_model["count"].sum()
    cluster_size_df_model["model_key"] = model_key

    kmeans_scores_ranked["model_key"] = model_key
    kmeans_scores_ranked["selected_k"] = int(best_k_model)
    kmeans_scores_ranked["auto_k_ranked"] = int(k_auto_model)
    kmeans_scores_ranked["selected_mode"] = k_mode_used
    kmeans_scores_ranked["n_rows_diag"] = int(len(X_diag))

    return {
        "embeddings": X_model,
        "best_k": best_k_model,
        "k_selection_mode": k_mode_used,
        "auto_k_ranked": int(k_auto_model),
        "labels": labels_model,
        "spectral_labels": spectral_labels_model,
        "kmeans_scores": kmeans_scores_ranked,
        "purity_table": purity_table_model,
        "stability_runs": stability_runs_model,
        "stability_summary": stability_summary_model,
        "cluster_size_df": cluster_size_df_model,
        "kmeans_final_metrics": kmeans_final_metrics_model,
        "spectral_metrics": spectral_metrics_model,
    }


def concat_by_field(phase3_by_model: dict, model_run_order: list[str], field_name: str) -> pd.DataFrame:
    """Concatenate phase3 result DataFrames for a given field name."""
    frames = [
        phase3_by_model[m][field_name]
        for m in model_run_order
        if m in phase3_by_model and isinstance(phase3_by_model[m].get(field_name), pd.DataFrame)
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

