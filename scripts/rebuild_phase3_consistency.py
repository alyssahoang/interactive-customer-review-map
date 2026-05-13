"""Repair Phase 3 artifacts so notebook, report tables, and run outputs are consistent.

This script rebuilds per-model Phase 3 summary files from canonical inputs:
- labels: data/phase3_multi/<model>/labels_kmeans.npy
- embeddings: data/embeddings/<model>_*_l2.npy
- labels/metadata: data/processed/reviews_preprocessed.pkl

It also refreshes run-level CSVs used by report pack generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


MODEL_KEYS = ("bertimbau", "multilingual_minilm", "openclip_text")
K_GRID = (2, 4, 6, 7, 8, 10, 12)
DIAG_SAMPLE_N = 4000
METRIC_SAMPLE_N = 12000
RANDOM_STATE = 42


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def stratified_indices(labels: np.ndarray, n_rows: int, sample_n: int, seed: int) -> np.ndarray:
    """Sample indices stratified by class labels."""
    if sample_n >= n_rows:
        return np.arange(n_rows, dtype=int)

    rng = np.random.default_rng(seed)
    y = pd.Series(labels).astype(str).reset_index(drop=True)
    picked_parts: list[np.ndarray] = []
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


def calc_purity(cluster_labels: np.ndarray, target_labels: np.ndarray) -> float:
    frame = pd.DataFrame({"cluster": cluster_labels, "target": target_labels})
    frame = frame[frame["cluster"] != -1]
    if frame.empty:
        return float("nan")
    dominant_counts = frame.groupby("cluster")["target"].value_counts().groupby(level=0).max()
    return float(dominant_counts.sum() / len(frame))


def run_k_diagnostics(X: np.ndarray, labels_for_strata: np.ndarray) -> tuple[pd.DataFrame, int]:
    """Compute K diagnostics on a stratified sample and return ranked table + auto-k."""
    from sklearn.cluster import KMeans

    diag_idx = stratified_indices(labels_for_strata, len(X), DIAG_SAMPLE_N, RANDOM_STATE)
    X_diag = X[diag_idx]

    rows = []
    for k in K_GRID:
        if k < 2 or k >= len(X_diag):
            continue
        km = KMeans(n_clusters=int(k), random_state=RANDOM_STATE, n_init=10)
        pred = km.fit_predict(X_diag)
        rows.append(
            {
                "k": int(k),
                "silhouette_cosine": float(silhouette_score(X_diag, pred, metric="cosine")),
                "calinski_harabasz": float(calinski_harabasz_score(X_diag, pred)),
                "davies_bouldin": float(davies_bouldin_score(X_diag, pred)),
                "inertia": float(km.inertia_),
            }
        )
    kdf = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    if kdf.empty:
        raise RuntimeError("K diagnostics produced no rows.")

    kdf["rank_sil"] = kdf["silhouette_cosine"].rank(ascending=False, method="min")
    kdf["rank_ch"] = kdf["calinski_harabasz"].rank(ascending=False, method="min")
    kdf["rank_db"] = kdf["davies_bouldin"].rank(ascending=True, method="min")
    kdf["rank_sum"] = kdf[["rank_sil", "rank_ch", "rank_db"]].sum(axis=1)
    auto_k = int(kdf.sort_values(["rank_sum", "k"]).iloc[0]["k"])
    kdf["n_rows_diag"] = int(len(X_diag))
    return kdf, auto_k


def build_topic_proxy(clean_text: pd.Series) -> np.ndarray:
    text_for_topic = clean_text.fillna("").astype(str).str.lower()
    return np.select(
        [
            text_for_topic.str.contains(r"\b(?:entrega|prazo|atras|frete|transport)\w*", regex=True),
            text_for_topic.str.contains(r"\b(?:produt|qualidad|defeit|quebrad|avari)\w*", regex=True),
            text_for_topic.str.contains(r"\b(?:atendiment|suport|sac|respost|contat)\w*", regex=True),
            text_for_topic.str.contains(r"\b(?:troc|devolu|reembols|cancel)\w*", regex=True),
            text_for_topic.str.contains(r"\b(?:preco|valor|car[oa]|barat|cust)\w*", regex=True),
        ],
        ["delivery", "product", "support", "refund", "price"],
        default="other",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild consistent Phase 3 artifacts.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "analysis-v1" / "20260511-234932",
        help="Run directory where report-pack input CSVs are refreshed.",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    submission_root = project_root / "submission"
    phase3_root = submission_root / "data" / "phase3_multi"
    run_dir = args.run_dir.resolve()

    reviews = pd.read_pickle(submission_root / "data" / "processed" / "reviews_preprocessed.pkl").reset_index(drop=True)
    n_rows = len(reviews)

    if "rating_group" in reviews.columns:
        sentiment_labels = reviews["rating_group"].fillna("neutral").astype(str).str.lower().to_numpy()
    elif "sentiment_label" in reviews.columns:
        sentiment_labels = reviews["sentiment_label"].fillna("neutral").astype(str).str.lower().to_numpy()
    else:
        sentiment_labels = np.array(["neutral"] * n_rows)

    topic_proxy = build_topic_proxy(reviews["clean_text"])

    emb_dir = project_root / "data" / "embeddings"

    phase3_rows: list[dict] = []
    kmeans_scores_all: list[pd.DataFrame] = []
    purity_all: list[pd.DataFrame] = []

    for model_key in MODEL_KEYS:
        model_dir = phase3_root / model_key
        labels_path = model_dir / "labels_kmeans.npy"
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing labels file: {labels_path}")
        labels = np.load(labels_path).astype(int)
        if len(labels) != n_rows:
            raise ValueError(f"{model_key}: labels length {len(labels)} != reviews length {n_rows}")

        l2_candidates = sorted(emb_dir.glob(f"{model_key}_n{n_rows}_d*_l2.npy"))
        if not l2_candidates:
            l2_candidates = sorted(emb_dir.glob(f"{model_key}_n*_d*_l2.npy"))
        if not l2_candidates:
            raise FileNotFoundError(f"No L2 embeddings found for model '{model_key}' in {emb_dir}")
        X = np.load(l2_candidates[-1]).astype(np.float32)
        if len(X) != n_rows:
            raise ValueError(f"{model_key}: embedding rows {len(X)} != reviews length {n_rows}")

        best_k = int(np.unique(labels).size)

        # Per-model cluster sizes from current labels
        cluster_sizes = (
            pd.Series(labels)
            .value_counts()
            .sort_index()
            .rename_axis("cluster_id")
            .reset_index(name="count")
        )
        cluster_sizes["ratio"] = cluster_sizes["count"] / float(cluster_sizes["count"].sum())
        cluster_sizes["model_key"] = model_key
        cluster_sizes.to_csv(model_dir / "cluster_sizes.csv", index=False)

        # Diagnostics table across k (always rebuilt to avoid stale/empty files)
        kdiag, auto_k = run_k_diagnostics(X, sentiment_labels)
        kdiag["model_key"] = model_key
        kdiag["selected_k"] = best_k
        kdiag["auto_k_ranked"] = auto_k
        kdiag["selected_mode"] = "fixed"
        kdiag.to_csv(model_dir / "kmeans_scores.csv", index=False)
        kmeans_scores_all.append(kdiag)

        # Main cluster metrics on a stratified metric sample for stability/speed
        metric_idx = stratified_indices(sentiment_labels, n_rows, METRIC_SAMPLE_N, RANDOM_STATE)
        X_eval = X[metric_idx]
        y_eval = labels[metric_idx]

        kmeans_final = pd.DataFrame(
            [
                {
                    "model_key": model_key,
                    "algorithm": "kmeans",
                    "k": best_k,
                    "silhouette_cosine": float(silhouette_score(X_eval, y_eval, metric="cosine")),
                    "calinski_harabasz": float(calinski_harabasz_score(X_eval, y_eval)),
                    "davies_bouldin": float(davies_bouldin_score(X_eval, y_eval)),
                    "inertia": float(np.nan),
                    "fit_rows": int(n_rows),
                    "eval_rows": int(len(metric_idx)),
                    "status": "recomputed_from_labels",
                }
            ]
        )
        kmeans_final.to_csv(model_dir / "kmeans_final_metrics.csv", index=False)

        purity_sent = calc_purity(labels, sentiment_labels)
        purity_topic = calc_purity(labels, topic_proxy)

        spectral_metrics = read_csv_or_empty(model_dir / "spectral_metrics.csv")
        spectral_sil = float(spectral_metrics.iloc[0]["silhouette_cosine"]) if not spectral_metrics.empty else float("nan")

        purity_tbl = pd.DataFrame(
            [
                {
                    "model_key": model_key,
                    "algorithm": "kmeans",
                    "purity_sentiment": purity_sent,
                    "purity_topic_proxy": purity_topic,
                }
            ]
        )

        # Preserve spectral row if already present for compatibility with existing notebook cells.
        old_purity = read_csv_or_empty(model_dir / "purity_table.csv")
        if not old_purity.empty and (old_purity["algorithm"].astype(str) == "spectral").any():
            spectral_row = old_purity[old_purity["algorithm"].astype(str) == "spectral"].copy()
            purity_tbl = pd.concat([purity_tbl, spectral_row], ignore_index=True)

        purity_tbl.to_csv(model_dir / "purity_table.csv", index=False)
        purity_all.append(purity_tbl)

        manifest = {
            "model_key": model_key,
            "best_k": int(best_k),
            "n_rows": int(n_rows),
        }
        (model_dir / "phase3_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        phase3_rows.append(
            {
                "model_key": model_key,
                "best_k": int(best_k),
                "k_mode": "fixed",
                "auto_k_ranked": int(auto_k),
                "kmeans_silhouette": float(kmeans_final.iloc[0]["silhouette_cosine"]),
                "spectral_silhouette": spectral_sil,
                "kmeans_purity_topic_proxy": float(purity_topic),
                "kmeans_purity_sentiment": float(purity_sent),
            }
        )

    phase3_cmp = pd.DataFrame(phase3_rows).sort_values("model_key").reset_index(drop=True)
    phase3_cmp.to_csv(phase3_root / "phase3_model_comparison.csv", index=False)

    kmeans_scores_all_df = pd.concat(kmeans_scores_all, ignore_index=True)

    # Refresh run-level CSVs consumed by report-prep script.
    run_dir.mkdir(parents=True, exist_ok=True)
    phase3_cmp.to_csv(run_dir / "phase3_model_comparison.csv", index=False)
    kmeans_scores_all_df.to_csv(run_dir / "kmeans_scores_all_models.csv", index=False)
    kmeans_scores_all_df.to_csv(run_dir / "kmeans_scores.csv", index=False)

    purity_all_df = pd.concat(purity_all, ignore_index=True)
    purity_all_df.to_csv(run_dir / "purity_table_all_models.csv", index=False)

    # Keep phase4 metrics from run_dir, but align purity columns with rebuilt Phase 3 values.
    phase4_path = run_dir / "phase4_model_comparison.csv"
    if phase4_path.exists():
        phase4_df = pd.read_csv(phase4_path)
        pmap_sent = dict(zip(phase3_cmp["model_key"], phase3_cmp["kmeans_purity_sentiment"]))
        pmap_topic = dict(zip(phase3_cmp["model_key"], phase3_cmp["kmeans_purity_topic_proxy"]))
        if "kmeans_purity_sentiment" in phase4_df.columns:
            phase4_df["kmeans_purity_sentiment"] = phase4_df["model_key"].map(pmap_sent)
        if "kmeans_purity_topic_proxy" in phase4_df.columns:
            phase4_df["kmeans_purity_topic_proxy"] = phase4_df["model_key"].map(pmap_topic)
        phase4_df.to_csv(phase4_path, index=False)

    rq_cmp_path = run_dir / "rq_model_comparison.csv"
    if rq_cmp_path.exists():
        rq_df = pd.read_csv(rq_cmp_path)
        pmap_sent = dict(zip(phase3_cmp["model_key"], phase3_cmp["kmeans_purity_sentiment"]))
        pmap_topic = dict(zip(phase3_cmp["model_key"], phase3_cmp["kmeans_purity_topic_proxy"]))
        if "kmeans_purity_sentiment" in rq_df.columns:
            rq_df["kmeans_purity_sentiment"] = rq_df["model_key"].map(pmap_sent)
        if "kmeans_purity_topic_proxy" in rq_df.columns:
            rq_df["kmeans_purity_topic_proxy"] = rq_df["model_key"].map(pmap_topic)
        rq_df.to_csv(rq_cmp_path, index=False)

    print("Rebuilt per-model Phase 3 artifacts and refreshed run-level comparison CSVs.")
    print(f"Phase 3 root: {phase3_root}")
    print(f"Run dir: {run_dir}")
    print(phase3_cmp.to_string(index=False))


if __name__ == "__main__":
    main()
