import argparse
from pathlib import Path
import json
import re
from collections import Counter
import sys
from textwrap import shorten

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluator import GeometryEvaluator


def _resolve_run_dir(run_root: Path, run_dir_arg: str | None) -> Path:
    required = [
        "kmeans_scores_all_models.csv",
        "phase3_model_comparison.csv",
        "topic_table.csv",
        "phase4_model_comparison.csv",
        "rq2_pairwise_delta_ci.csv",
    ]

    if run_dir_arg:
        run_dir = Path(run_dir_arg).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Explicit run-dir does not exist: {run_dir}")
        return run_dir

    if not run_root.exists():
        raise FileNotFoundError(f"Run root not found: {run_root}")

    candidates = [p for p in run_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under: {run_root}")

    scored: list[tuple[int, str, Path]] = []
    for run in candidates:
        score = sum(int((run / name).exists()) for name in required)
        scored.append((score, run.name, run))

    scored.sort(key=lambda x: (x[0], x[1]))
    best_score, _, best_run = scored[-1]
    if best_score == 0:
        raise FileNotFoundError(
            "No usable run folder found. Expected run files were not detected under "
            f"{run_root}. Run notebook export cells first."
        )
    return best_run


def _load_optional_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path)
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    return df[columns].copy()


parser = argparse.ArgumentParser(description="Prepare report output tables/figures from cached run artifacts.")
parser.add_argument("--run-root", type=Path, default=ROOT / "artifacts" / "analysis-v1")
parser.add_argument("--run-dir", type=str, default=None, help="Optional explicit run directory path.")
parser.add_argument("--out-dir", type=Path, default=ROOT / "report" / "analysis-v1-v2")
args = parser.parse_args()

RUN_ROOT = args.run_root.resolve()
RUN_DIR = _resolve_run_dir(RUN_ROOT, args.run_dir)

OUT_DIR = args.out_dir.resolve()
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

print(f"Using run folder: {RUN_DIR}")

model_order = ["bertimbau", "multilingual_minilm", "openclip_text"]
model_label = {
    "bertimbau": "BERTimbau",
    "multilingual_minilm": "MiniLM",
    "openclip_text": "OpenCLIP",
}


def _resolve_embedding_root() -> Path | None:
    """Find the embeddings directory, preferring submission-local assets."""
    candidates = [
        ROOT / "data" / "embeddings",
        ROOT.parent / "data" / "embeddings",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _pick_l2_embedding(embedding_root: Path, model_key: str, n_rows: int) -> Path | None:
    """Pick an L2 embedding file for model_key, prioritizing matching n_rows."""
    candidates = sorted(embedding_root.glob(f"{model_key}_*_l2.npy"))
    if not candidates:
        return None
    matched = [p for p in candidates if f"_n{n_rows}_" in p.name]
    if matched:
        return matched[-1]
    return candidates[-1]

reviews = pd.read_pickle(ROOT / "data" / "processed" / "reviews_preprocessed.pkl").reset_index(drop=True)
reviews["review_text"] = reviews["review_text"].fillna("").astype(str)
reviews["clean_text"] = reviews["clean_text"].fillna("").astype(str)
if "review_id" not in reviews.columns:
    reviews["review_id"] = np.arange(len(reviews), dtype=int)

reviews_en = pd.read_csv(
    ROOT / "data" / "olist_order_reviews_dataset_translated.csv",
    usecols=["review_id", "review_comment_message"],
)
reviews_en = reviews_en.drop_duplicates(subset=["review_id"], keep="first")
reviews_en = reviews_en.rename(columns={"review_comment_message": "review_text_en"})
reviews = reviews.merge(reviews_en, on="review_id", how="left")
reviews["review_text_en"] = reviews["review_text_en"].fillna("").astype(str)

# Correct a known untranslated fragment in the source translation file
# so figure labels stay consistently in English.
manual_en_overrides = {
    "436e9da2a75eb6d50db3cac5037a81a6": "The product was not delivered completely; the main item was missing.",
}
if "review_id" in reviews.columns:
    override_mask = reviews["review_id"].isin(manual_en_overrides)
    reviews.loc[override_mask, "review_text_en"] = (
        reviews.loc[override_mask, "review_id"].map(manual_en_overrides).fillna("")
    )

sent_raw = reviews["sentiment_label"].astype(str).str.strip().str.lower()
sent_map = {"negative": "negative", "neutral": "neutral", "positive": "positive"}
reviews["sentiment_for_report"] = sent_raw.map(sent_map)
if reviews["sentiment_for_report"].isna().all() and "rating_group" in reviews.columns:
    reviews["sentiment_for_report"] = reviews["rating_group"].astype(str).str.lower().map(sent_map)
reviews["sentiment_for_report"] = reviews["sentiment_for_report"].fillna("neutral")

txt = reviews["clean_text"].str.lower()
reviews["topic_proxy"] = np.select(
    [
        txt.str.contains(r"\b(?:entrega|entregou|prazo|atras|transportadora|frete|chegou)\b", regex=True),
        txt.str.contains(r"\b(?:produto|qualidade|defeito|material|tamanho|cor|quebrado)\b", regex=True),
        txt.str.contains(r"\b(?:atendimento|suporte|resposta|sac|vendedor|contato)\b", regex=True),
        txt.str.contains(r"\b(?:reembolso|devolu|troca|cancelamento|estorno)\b", regex=True),
        txt.str.contains(r"\b(?:preco|preço|valor|caro|barato|custo|desconto)\b", regex=True),
    ],
    ["delivery", "product", "support", "refund", "price"],
    default="other",
)

# ------------------------------ Phase 1 ------------------------------
rating = reviews["rating_group"].astype(str).str.lower()
lexicon = reviews["lexicon_group"].astype(str).str.lower()
order = ["negative", "neutral", "positive"]

cm_counts = pd.crosstab(
    pd.Categorical(rating, categories=order),
    pd.Categorical(lexicon, categories=order),
    dropna=False,
).fillna(0).astype(int)
cm_counts.index.name = "rating_group"
cm_counts.columns.name = "lexicon_group"
cm_rows = cm_counts.div(cm_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

cm_counts.to_csv(TABLE_DIR / "phase1_rating_vs_lexicon_counts.csv")
cm_rows.to_csv(TABLE_DIR / "phase1_rating_vs_lexicon_row_normalized.csv")

fig, ax = plt.subplots(figsize=(6.8, 5.2))
sns.heatmap(cm_counts, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_title("Rating vs Lexicon (Counts)")
fig.tight_layout()
fig.savefig(FIG_DIR / "phase1_rating_vs_lexicon_counts_heatmap.png", dpi=220)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.8, 5.2))
sns.heatmap(cm_rows, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1, ax=ax)
ax.set_title("Rating vs Lexicon (Row-normalized)")
fig.tight_layout()
fig.savefig(FIG_DIR / "phase1_rating_vs_lexicon_rownorm_heatmap.png", dpi=220)
plt.close(fig)

overall_agreement = float((rating == lexicon).mean())
if "label_agree" in reviews.columns:
    overall_agreement = float(reviews["label_agree"].astype(bool).mean())

agreement_df = pd.DataFrame(
    [
        {"metric": "overall_agreement", "value": overall_agreement},
        {
            "metric": "polite_negative_rate",
            "value": float(reviews["polite_negative_flag"].astype(bool).mean())
            if "polite_negative_flag" in reviews.columns
            else np.nan,
        },
        {
            "metric": "mixed_positive_rate",
            "value": float(reviews["mixed_positive_flag"].astype(bool).mean())
            if "mixed_positive_flag" in reviews.columns
            else np.nan,
        },
        {"metric": "n_reviews", "value": int(len(reviews))},
    ]
)
agreement_df.to_csv(TABLE_DIR / "phase1_agreement_metrics.csv", index=False)

# ------------------------------ Phase 2 ------------------------------
embedding_dir = ROOT / "data" / "embeddings"
norm_rows = []
if embedding_dir.exists():
    for mk in model_order:
        raw_candidates = sorted(embedding_dir.glob(f"{mk}_*_raw.npy"))
        l2_candidates = sorted(embedding_dir.glob(f"{mk}_*_l2.npy"))
        if not raw_candidates or not l2_candidates:
            continue

        raw_path = raw_candidates[-1]
        l2_path = l2_candidates[-1]

        raw = np.load(raw_path, mmap_mode="r")
        l2 = np.load(l2_path, mmap_mode="r")

        raw_norm = np.linalg.norm(raw, axis=1)
        l2_norm = np.linalg.norm(l2, axis=1)

        norm_rows.append(
            {
                "model_key": mk,
                "model_label": model_label[mk],
                "n_rows": int(raw.shape[0]),
                "dim": int(raw.shape[1]),
                "raw_norm_mean": float(raw_norm.mean()),
                "raw_norm_std": float(raw_norm.std()),
                "l2_norm_mean": float(l2_norm.mean()),
                "l2_norm_std": float(l2_norm.std()),
                "raw_file": raw_path.name,
                "l2_file": l2_path.name,
            }
        )
else:
    print("Warning: data/embeddings not found; Phase 2 norm table will be empty.")

embedding_norm_df = pd.DataFrame(norm_rows)
embedding_norm_df.to_csv(TABLE_DIR / "phase2_embedding_norms.csv", index=False)

# ------------------------------ Phase 3 ------------------------------
kmeans_scores = pd.read_csv(RUN_DIR / "kmeans_scores_all_models.csv")
kmeans_scores["model_label"] = kmeans_scores["model_key"].map(model_label)
kmeans_scores.to_csv(TABLE_DIR / "phase3_kmeans_kcurve.csv", index=False)

fig, ax = plt.subplots(figsize=(8.8, 5.2))
sns.lineplot(
    data=kmeans_scores,
    x="k",
    y="silhouette_cosine",
    hue="model_label",
    marker="o",
    ax=ax,
)
ax.set_title("K-Means diagnostics: silhouette vs k")
ax.set_ylabel("Silhouette (cosine)")
fig.tight_layout()
fig.savefig(FIG_DIR / "phase3_kmeans_kcurve_silhouette.png", dpi=230)
plt.close(fig)

phase3_cmp = pd.read_csv(RUN_DIR / "phase3_model_comparison.csv")
phase3_cmp["model_label"] = phase3_cmp["model_key"].map(model_label)
phase3_cmp.to_csv(TABLE_DIR / "phase3_model_comparison.csv", index=False)

cluster_rows = []
for mk in model_order:
    p = ROOT / "data" / "phase3_multi" / mk / "cluster_sizes.csv"
    if p.exists():
        d = pd.read_csv(p)
        d["model_key"] = mk
        d["model_label"] = model_label[mk]
        cluster_rows.append(d)

cluster_size_df = pd.concat(cluster_rows, ignore_index=True)
cluster_size_df.to_csv(TABLE_DIR / "phase3_cluster_size_distribution_k7.csv", index=False)

fig, ax = plt.subplots(figsize=(10.4, 5.4))
sns.barplot(
    data=cluster_size_df,
    x="cluster_id",
    y="ratio",
    hue="model_label",
    ax=ax,
)
ax.set_title("Cluster size distribution at k=7")
ax.set_ylabel("Cluster ratio")
fig.tight_layout()
fig.savefig(FIG_DIR / "phase3_cluster_size_distribution_k7.png", dpi=230)
plt.close(fig)

topic_map = reviews.set_index("review_id")["topic_proxy"].to_dict()
sent_palette = {"negative": "#d73027", "neutral": "#7f7f7f", "positive": "#1a9850"}
topic_palette = {
    "delivery": "#1f77b4",
    "product": "#2ca02c",
    "support": "#ff7f0e",
    "refund": "#d62728",
    "price": "#9467bd",
    "other": "#8c8c8c",
}
sent_style_order = ["positive", "negative", "neutral"]
# Use simple geometric markers to keep dense maps readable.
sent_marker_map = {
    "positive": "o",
    "negative": "X",
    "neutral": "^",
}
proj_figsize = (21.5, 6.6)
proj_marker_size = 10
proj_linewidth = 0.12
proj_dpi = 520
phase3_dir = ROOT / "data" / "phase3_multi"
projection_dir = phase3_dir / "projection_views"


def _load_projection_viz(method_key: str) -> pd.DataFrame:
    """Prefer full per-model coordinates; fallback to legacy bundled viz files."""
    full_rows = []
    for mk in model_order:
        coords_path = phase3_dir / mk / f"coords_{method_key}.npy"
        labels_path = phase3_dir / mk / "label_index.parquet"
        if not (coords_path.exists() and labels_path.exists()):
            continue
        coords = np.load(coords_path)
        labels = pd.read_parquet(labels_path)
        if not {"review_id", "kmeans_cluster"}.issubset(labels.columns):
            continue
        n = min(len(coords), len(labels))
        labels = labels.iloc[:n].copy()
        part = pd.DataFrame(
            {
                "x": coords[:n, 0],
                "y": coords[:n, 1],
                "cluster": labels["kmeans_cluster"].astype(str),
                "review_id": labels["review_id"].astype(str),
                "model_key": mk,
            }
        )
        if "sentiment_label" in labels.columns:
            part["sentiment"] = labels["sentiment_label"].astype(str).str.lower().map(sent_map).fillna("neutral")
        else:
            part["sentiment"] = "neutral"
        if "topic_proxy" in labels.columns:
            part["topic_proxy"] = labels["topic_proxy"].astype(str).str.lower().fillna("other")
        else:
            part["topic_proxy"] = part["review_id"].map(topic_map).fillna("other")
        full_rows.append(part)

    if full_rows:
        return pd.concat(full_rows, ignore_index=True)

    for ext in ("parquet", "csv"):
        p = projection_dir / f"viz_{method_key}.{ext}"
        if not p.exists():
            continue
        viz = pd.read_parquet(p).copy() if ext == "parquet" else pd.read_csv(p).copy()
        if "topic_proxy" not in viz.columns:
            viz["topic_proxy"] = viz["review_id"].map(topic_map).fillna("other")
        viz["sentiment"] = viz["sentiment"].astype(str).str.lower().map(sent_map).fillna("neutral")
        return viz

    return pd.DataFrame()


for method_key, method_name in [("pca", "PCA"), ("tsne", "tSNE"), ("umap", "UMAP")]:
    viz = _load_projection_viz(method_key)
    if viz.empty:
        continue

    fig, axes = plt.subplots(1, 3, figsize=proj_figsize, sharex=False, sharey=False)
    for ax, mk in zip(axes, model_order):
        d = viz[viz["model_key"] == mk]
        sns.scatterplot(
            data=d,
            x="x",
            y="y",
            hue="cluster",
            palette="tab20",
            s=proj_marker_size,
            linewidth=proj_linewidth,
            alpha=0.62,
            rasterized=True,
            legend=False,
            ax=ax,
        )
        ax.set_title(f"{model_label[mk]} - {method_name} (cluster)")
        ax.set_xlabel("component 1")
        ax.set_ylabel("component 2")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"phase3_projection_{method_key}_cluster.png", dpi=proj_dpi, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"phase3_projection_{method_key}_cluster.pdf", dpi=proj_dpi, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=proj_figsize, sharex=False, sharey=False)
    for j, (ax, mk) in enumerate(zip(axes, model_order)):
        d = viz[viz["model_key"] == mk]
        sns.scatterplot(
            data=d,
            x="x",
            y="y",
            hue="sentiment",
            hue_order=["negative", "neutral", "positive"],
            palette=sent_palette,
            s=proj_marker_size,
            linewidth=proj_linewidth,
            alpha=0.62,
            rasterized=True,
            legend=j == 0,
            ax=ax,
        )
        if j == 0:
            leg = ax.get_legend()
            if leg is not None:
                leg.set_title("sentiment")
                handles = getattr(leg, "legend_handles", None)
                if handles is None:
                    handles = getattr(leg, "legendHandles", [])
                for h in handles:
                    if hasattr(h, "set_sizes"):
                        h.set_sizes([90])
                    if hasattr(h, "set_alpha"):
                        h.set_alpha(1.0)
                    if hasattr(h, "set_edgecolor"):
                        h.set_edgecolor("black")
                    if hasattr(h, "set_linewidth"):
                        h.set_linewidth(0.55)
        ax.set_title(f"{model_label[mk]} - {method_name} (sentiment)")
        ax.set_xlabel("component 1")
        ax.set_ylabel("component 2")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"phase3_projection_{method_key}_sentiment.png", dpi=proj_dpi, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"phase3_projection_{method_key}_sentiment.pdf", dpi=proj_dpi, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=proj_figsize, sharex=False, sharey=False)
    for ax, mk in zip(axes, model_order):
        d = viz[viz["model_key"] == mk]
        sns.scatterplot(
            data=d,
            x="x",
            y="y",
            hue="cluster",
            style="sentiment",
            style_order=sent_style_order,
            markers=sent_marker_map,
            palette="tab20",
            s=proj_marker_size,
            linewidth=proj_linewidth,
            alpha=0.6,
            legend=False,
            ax=ax,
        )
        ax.set_title(f"{model_label[mk]} - {method_name} (cluster + sentiment)")
        ax.set_xlabel("component 1")
        ax.set_ylabel("component 2")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"phase3_projection_{method_name.lower()}_overlay_cluster_sentiment.png", dpi=proj_dpi, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"phase3_projection_{method_name.lower()}_overlay_cluster_sentiment.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=proj_figsize, sharex=False, sharey=False)
    for j, (ax, mk) in enumerate(zip(axes, model_order)):
        d = viz[viz["model_key"] == mk]
        sns.scatterplot(
            data=d,
            x="x",
            y="y",
            hue="topic_proxy",
            style="sentiment",
            style_order=sent_style_order,
            markers=sent_marker_map,
            hue_order=["delivery", "product", "support", "refund", "price", "other"],
            palette=topic_palette,
            s=proj_marker_size,
            linewidth=proj_linewidth,
            alpha=0.6,
            legend=j == 0,
            ax=ax,
        )
        ax.set_title(f"{model_label[mk]} - {method_name} (topic proxy + sentiment)")
        ax.set_xlabel("component 1")
        ax.set_ylabel("component 2")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"phase3_projection_{method_name.lower()}_overlay_topic_sentiment.png", dpi=proj_dpi, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"phase3_projection_{method_name.lower()}_overlay_topic_sentiment.pdf", bbox_inches="tight")
    plt.close(fig)

proj_quality = pd.read_csv(ROOT / "data" / "phase3_multi" / "projection_quality_all_models.csv")
proj_quality["model_label"] = proj_quality["model_key"].map(model_label)
proj_quality.to_csv(TABLE_DIR / "phase3_projection_quality.csv", index=False)

fig, ax = plt.subplots(figsize=(9.3, 5.2))
sns.barplot(
    data=proj_quality,
    x="projection",
    y="silhouette_topic_proxy",
    hue="model_label",
    ax=ax,
)
ax.set_title("2D projection quality: topic silhouette")
fig.tight_layout()
fig.savefig(FIG_DIR / "phase3_projection_quality_topic_silhouette.png", dpi=230)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9.3, 5.2))
sns.barplot(
    data=proj_quality,
    x="projection",
    y="delta_topic_minus_sentiment",
    hue="model_label",
    ax=ax,
)
ax.axhline(0, color="black", linestyle="--", linewidth=1)
ax.set_title("2D projection quality: delta(topic - sentiment)")
fig.tight_layout()
fig.savefig(FIG_DIR / "phase3_projection_quality_delta_topic_minus_sentiment.png", dpi=230)
plt.close(fig)

knn_summary_path = TABLE_DIR / "phase3_knn_anchor_ambiguous_k20_summary.csv"
l2_available = all(sorted((ROOT / "data" / "embeddings").glob(f"{mk}_*_l2.npy")) for mk in model_order)
if knn_summary_path.exists() and l2_available:
    knn_summary = pd.read_csv(knn_summary_path)
    anchor_idx = int(knn_summary.loc[0, "anchor_idx"])
    k_knn = int(knn_summary.loc[0, "k"])
    sent_colors = {"negative": "#d73027", "neutral": "#7f7f7f", "positive": "#1a9850"}

    fig, axes = plt.subplots(1, 3, figsize=(22, 7.2), sharex=False, sharey=False)

    for col, (ax, mk) in enumerate(zip(axes, model_order)):
        emb_path = sorted((ROOT / "data" / "embeddings").glob(f"{mk}_*_l2.npy"))[-1]
        emb = np.load(emb_path, mmap_mode="r")

        sims = emb @ emb[anchor_idx]
        rank_idx = np.argsort(-sims)
        neighbor_idx = rank_idx[rank_idx != anchor_idx][:k_knn]
        local_idx = np.r_[anchor_idx, neighbor_idx]
        coords = PCA(n_components=2, random_state=42).fit_transform(emb[local_idx])

        panel = pd.DataFrame(
            {
                "idx": local_idx,
                "x": coords[:, 0],
                "y": coords[:, 1],
                "sentiment": reviews.loc[local_idx, "sentiment_for_report"].to_numpy(),
                "snippet_en": reviews.loc[local_idx, "review_text_en"].to_numpy(),
                "role": ["anchor"] + ["neighbor"] * len(neighbor_idx),
            }
        )
        panel["rank"] = 0
        panel.loc[panel["role"] == "neighbor", "rank"] = np.arange(1, len(neighbor_idx) + 1)

        neighbors = panel[panel["role"] == "neighbor"].copy()
        for sent in ["negative", "neutral", "positive"]:
            d = neighbors[neighbors["sentiment"] == sent]
            ax.scatter(
                d["x"],
                d["y"],
                s=38,
                alpha=0.85,
                color=sent_colors[sent],
                edgecolor="white",
                linewidth=0.35,
                label=sent.capitalize() if col == 0 else None,
            )

        anchor = panel[panel["role"] == "anchor"].iloc[0]
        ax.scatter(
            [anchor["x"]],
            [anchor["y"]],
            marker="*",
            s=350,
            color="#1abc9c",
            edgecolor="black",
            linewidth=1.0,
            label="Anchor" if col == 0 else None,
            zorder=4,
        )

        top_n_labels = 8
        label_offsets = [
            (10, 10),
            (10, -10),
            (-10, 10),
            (-10, -10),
            (14, 4),
            (14, -4),
            (-14, 4),
            (-14, -4),
        ]
        labels_drawn = 0
        for _, row in neighbors.nsmallest(k_knn, "rank").iterrows():
            if labels_drawn >= top_n_labels:
                break
            snippet = str(row["snippet_en"]).replace("\n", " ").strip()
            if not snippet:
                continue
            label = shorten(snippet, width=34, placeholder="...")
            dx, dy = label_offsets[labels_drawn % len(label_offsets)]
            ax.annotate(
                label,
                xy=(row["x"], row["y"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=6.8,
                ha="left" if dx >= 0 else "right",
                va="bottom" if dy >= 0 else "top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#888888", lw=0.5, alpha=0.92),
                arrowprops=dict(arrowstyle="-", color="#888888", lw=0.5, alpha=0.7),
                zorder=5,
            )
            labels_drawn += 1

        anchor_text = str(anchor["snippet_en"]).replace("\n", " ").strip()
        if not anchor_text:
            anchor_text = "No English translation available."
        ax.annotate(
            f"A: {shorten(anchor_text, width=34, placeholder='...')}",
            xy=(anchor["x"], anchor["y"]),
            xytext=(12, 12),
            textcoords="offset points",
            fontsize=7.0,
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.22", fc="#eafff8", ec="#16a085", lw=0.7, alpha=0.95),
            arrowprops=dict(arrowstyle="-", color="#16a085", lw=0.6, alpha=0.8),
            zorder=6,
        )

        ax.set_title(f"({chr(97 + col)}) {model_label[mk]}", fontsize=15, pad=8)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.25)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=sent_colors["negative"], markersize=8, label="Negative"),
        plt.Line2D([0], [0], marker="o", linestyle="", color=sent_colors["neutral"], markersize=8, label="Neutral"),
        plt.Line2D([0], [0], marker="o", linestyle="", color=sent_colors["positive"], markersize=8, label="Positive"),
        plt.Line2D([0], [0], marker="*", linestyle="", markerfacecolor="#1abc9c", markeredgecolor="black", markersize=14, label="Anchor"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=11)
    fig.suptitle(
        "Local k-NN neighbourhoods (k=20) around the same ambiguous anchor with inline English snippets",
        fontsize=14,
        y=1.06,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(FIG_DIR / "phase3_knn_neighbourhoods_anchor_ambiguous_k20_en.png", dpi=280, bbox_inches="tight")
    plt.close(fig)
else:
    print("Warning: skipping kNN neighbourhood figure (missing summary table and/or L2 embeddings).")

# ------------------------------ Phase 4 ------------------------------
topic_table = pd.read_csv(RUN_DIR / "topic_table.csv")
topic_table["model_key"] = topic_table["model_key"].astype(str)
topic_table["cluster_id"] = topic_table["cluster_id"].astype(int)
topic_lookup = topic_table.set_index(["model_key", "cluster_id"])["topic_name_keyword"].to_dict()

stop_tokens = {
    "muito",
    "produto",
    "prazo",
    "entrega",
    "foi",
    "para",
    "com",
    "que",
    "uma",
    "não",
    "mais",
    "bom",
    "boa",
    "otimo",
    "ótimo",
}
cluster_rows = []
for mk in model_order:
    labels_path = ROOT / "data" / "phase3_multi" / mk / "labels_kmeans.npy"
    if not labels_path.exists():
        continue
    labels = np.load(labels_path).astype(int)

    for cid in sorted(np.unique(labels)):
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            continue

        neg_ratio = float((reviews["sentiment_for_report"].iloc[idx] == "negative").mean())
        size = int(len(idx))

        theme = topic_lookup.get((mk, int(cid)), "")
        theme_source = "topic_name_keyword"
        if not isinstance(theme, str) or not theme.strip():
            tokens = " ".join(reviews["clean_text"].iloc[idx].astype(str).tolist()).lower().split()
            tokens = [t for t in tokens if len(t) >= 4 and t.isalpha() and t not in stop_tokens]
            top_terms = [w for w, _ in Counter(tokens).most_common(5)]
            theme = " | ".join(top_terms) if top_terms else "other"
            theme_source = "auto_keyword"

        review_series = reviews["review_text"].iloc[idx].astype(str)
        review_series = review_series[review_series.str.len() > 15]
        review_series = review_series.sort_values(key=lambda s: s.str.len(), ascending=False).head(2)
        ex1 = review_series.iloc[0] if len(review_series) > 0 else ""
        ex2 = review_series.iloc[1] if len(review_series) > 1 else ""

        cluster_rows.append(
            {
                "model_key": mk,
                "model_label": model_label[mk],
                "cluster_id": int(cid),
                "size": size,
                "size_share": size / len(reviews),
                "negative_ratio": neg_ratio,
                "final_theme": theme,
                "theme_source": theme_source,
                "example_review_1": ex1,
                "example_review_2": ex2,
            }
        )

cluster_profile_df = pd.DataFrame(cluster_rows).sort_values(["model_key", "cluster_id"]).reset_index(drop=True)
cluster_profile_df.to_csv(TABLE_DIR / "phase4_cluster_profile_all_models.csv", index=False)

selected_model = topic_table["model_key"].mode().iloc[0] if len(topic_table) else "multilingual_minilm"
prototypes = _load_optional_csv(
    RUN_DIR / "cluster_prototypes.csv",
    columns=["cluster_id", "prototype_text"],
)
if "model_key" not in prototypes.columns:
    prototypes.insert(0, "model_key", selected_model)
else:
    prototypes["model_key"] = prototypes["model_key"].fillna(selected_model)
prototypes.to_csv(TABLE_DIR / "phase4_cluster_prototypes_selected_model.csv", index=False)

counterfactual_cols = [
    "source_idx",
    "counterfactual_idx",
    "source_sentiment",
    "counterfactual_sentiment",
    "source_cluster",
    "counterfactual_cluster",
    "cosine_similarity",
    "source_text",
    "counterfactual_text",
    "source_text_en",
    "counterfactual_text_en",
]
counterfactual = _load_optional_csv(RUN_DIR / "nearest_counterfactuals.csv", columns=counterfactual_cols)
counterfactual.to_csv(TABLE_DIR / "phase4_counterfactual_pairs.csv", index=False)

evaluator = GeometryEvaluator(random_state=42)
neg_profile_rows = []
embedding_root = _resolve_embedding_root()
if embedding_root is not None:
    print(f"Using embeddings from: {embedding_root}")
    for mk in model_order:
        emb_path = _pick_l2_embedding(embedding_root, mk, n_rows=len(reviews))
        labels_path = ROOT / "data" / "phase3_multi" / mk / "labels_kmeans.npy"
        if emb_path is None or not labels_path.exists():
            continue
        labels = np.load(labels_path).astype(int)
        emb = np.load(emb_path, mmap_mode="r")
        prof = evaluator.negative_profile(
            embeddings=emb,
            labels=labels,
            sentiments=reviews["sentiment_for_report"],
        ).copy()
        prof["model_key"] = mk

        global_neg = float((reviews["sentiment_for_report"] == "negative").mean())
        se = np.sqrt(np.clip((global_neg * (1.0 - global_neg)) / prof["size"].to_numpy(), 1e-12, None))
        prof["z_negative"] = (prof["negative_ratio"].to_numpy() - global_neg) / se
        density_cut = float(prof["dispersion"].quantile(0.40))
        prof["dense_flag"] = prof["dispersion"] <= density_cut
        prof["island_flag"] = (prof["z_negative"] >= 2.0) & prof["dense_flag"]
        neg_profile_rows.append(prof)

if not neg_profile_rows:
    fallback_path = RUN_DIR / "negative_profile.csv"
    if fallback_path.exists():
        neg_profile_rows.append(pd.read_csv(fallback_path))

if neg_profile_rows:
    neg_profile_all = pd.concat(neg_profile_rows, ignore_index=True)
else:
    neg_profile_all = pd.DataFrame(
        columns=[
            "cluster_id",
            "size",
            "mean_cosine_similarity_to_centroid",
            "dispersion",
            "negative_ratio",
            "model_key",
            "z_negative",
            "dense_flag",
            "island_flag",
        ]
    )
neg_profile_all["model_label"] = neg_profile_all["model_key"].map(model_label)
neg_profile_all.to_csv(TABLE_DIR / "phase4_negative_profile_all_models.csv", index=False)

if not neg_profile_all.empty:
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.3), sharey=True)
    for ax, mk in zip(axes, model_order):
        d = neg_profile_all[neg_profile_all["model_key"] == mk].copy()
        d = d.sort_values("cluster_id")
        colors = d["island_flag"].map({True: "#d73027", False: "#9e9e9e"}).tolist()
        ax.bar(d["cluster_id"].astype(str), d["negative_ratio"], color=colors)
        ax.set_title(f"{model_label[mk]}: negative ratio by cluster")
        ax.set_xlabel("cluster_id")
        ax.set_ylabel("negative_ratio")
        ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase4_negative_ratio_by_cluster_island_flag.png", dpi=240)
    plt.close(fig)
else:
    print("Warning: negative-profile figure skipped (no profile rows available).")

phase4_cmp = pd.read_csv(RUN_DIR / "phase4_model_comparison.csv")
phase4_cmp["model_label"] = phase4_cmp["model_key"].map(model_label)
phase4_cmp.to_csv(TABLE_DIR / "phase4_model_comparison.csv", index=False)

rq2_ci = pd.read_csv(RUN_DIR / "rq2_pairwise_delta_ci.csv")
rq2_ci.to_csv(TABLE_DIR / "phase4_rq2_pairwise_delta_ci.csv", index=False)

# ------------------------------ Report shell ------------------------------
rq_rows = phase4_cmp[
    [
        "model_label",
        "cluster_negative_ratio_std",
        "negative_island_count",
        "negative_island_mass",
        "rq_signal",
    ]
].sort_values("cluster_negative_ratio_std", ascending=False)

structure_md = f"""# P15 Report Output Pack (analysis-v1-v2)

Source run: `{RUN_DIR.name}`  
Prepared on: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}

## 1. Introduction
- Motivation: star ratings are coarse; review text carries richer semantic signals.
- RQ1: Do negative reviews form geometric islands or a diffuse layer over topics?
- RQ2: How does islandization differ across BERTimbau, MiniLM, OpenCLIP?

## 2. Data
- Dataset rows after preprocessing: **{len(reviews):,}**
- Use:
  - `tables/phase1_rating_vs_lexicon_counts.csv`
  - `tables/phase1_rating_vs_lexicon_row_normalized.csv`
  - `tables/phase1_agreement_metrics.csv`
  - `figures/phase1_rating_vs_lexicon_counts_heatmap.png`
  - `figures/phase1_rating_vs_lexicon_rownorm_heatmap.png`

## 3. Methods
- Embedding norms:
  - `tables/phase2_embedding_norms.csv`
- Geometry:
  - `tables/phase3_kmeans_kcurve.csv`
  - `figures/phase3_kmeans_kcurve_silhouette.png`
  - `tables/phase3_model_comparison.csv`
  - `tables/phase3_cluster_size_distribution_k7.csv`
  - `figures/phase3_cluster_size_distribution_k7.png`
  - Projection maps by method:
    - cluster: `figures/phase3_projection_*_cluster.png`
    - sentiment: `figures/phase3_projection_*_sentiment.png`
    - overlays: `figures/phase3_projection_*_overlay_cluster_sentiment.png`
    - topic/sentiment overlays: `figures/phase3_projection_*_overlay_topic_sentiment.png`
  - Projection quality:
    - `tables/phase3_projection_quality.csv`
    - `figures/phase3_projection_quality_topic_silhouette.png`
    - `figures/phase3_projection_quality_delta_topic_minus_sentiment.png`

## 4. Results
- Phase 4 tables:
  - `tables/phase4_cluster_profile_all_models.csv`
  - `tables/phase4_cluster_prototypes_selected_model.csv`
  - `tables/phase4_counterfactual_pairs.csv`
  - `tables/phase4_negative_profile_all_models.csv`
  - `figures/phase4_negative_ratio_by_cluster_island_flag.png`
  - `tables/phase4_model_comparison.csv`
  - `tables/phase4_rq2_pairwise_delta_ci.csv`

### Quick RQ snapshot
{rq_rows.to_string(index=False)}

## 5. Discussion
- Interpret whether negative regions are topic-bound islands or diffuse modifiers.
- Contrast MiniLM diffusion vs BERTimbau/OpenCLIP islandization patterns using the phase4 comparison and pairwise CI tables.
- Note limitations: lexicon baseline, regex topic proxy, and local LLM naming variability.
"""
(OUT_DIR / "REPORT_DRAFT_STRUCTURE.md").write_text(structure_md, encoding="utf-8")

manifest_rows = []
for p in sorted(OUT_DIR.rglob("*")):
    if p.is_file():
        manifest_rows.append(
            {
                "relative_path": str(p.relative_to(OUT_DIR)),
                "size_bytes": p.stat().st_size,
            }
        )
pd.DataFrame(manifest_rows).to_csv(OUT_DIR / "manifest.csv", index=False)

print(f"Saved report output pack to: {OUT_DIR}")
