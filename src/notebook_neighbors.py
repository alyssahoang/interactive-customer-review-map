"""Notebook helpers for local semantic-neighborhood inspection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from .notebook_text_helpers import to_en_safe


def inspect_neighbors(
    anchor_idx: int,
    reviews_df: pd.DataFrame,
    embeddings_l2_by_model: dict[str, np.ndarray],
    top_k: int = 5,
) -> tuple[pd.DataFrame, str]:
    """Inspect top-k cosine neighbors across models for one anchor review."""
    anchor = reviews_df.iloc[anchor_idx]
    anchor_sent = anchor["sentiment_label"]

    print("=" * 90)
    print(f"Anchor idx: {anchor_idx} | sentiment: {anchor_sent}")
    print("PT:", anchor["review_text"])
    anchor_review_id = str(anchor["review_id"]) if "review_id" in reviews_df.columns else None
    print("EN:", to_en_safe(anchor["review_text"], review_id=anchor_review_id))
    print("=" * 90)

    cols = [
        "model_key",
        "rank",
        "neighbor_idx",
        "cosine_similarity",
        "sentiment",
        "review_text",
        "snippet",
        "snippet_en",
        "same_sentiment",
    ]
    if not embeddings_l2_by_model:
        return pd.DataFrame(columns=cols), anchor_sent

    neighbor_rows: list[dict] = []
    for model_key, matrix in embeddings_l2_by_model.items():
        sims = cosine_similarity(matrix[anchor_idx : anchor_idx + 1], matrix).ravel()
        nearest_idx = np.argsort(sims)[::-1][1 : top_k + 1]
        for rank, idx in enumerate(nearest_idx, start=1):
            text_pt = reviews_df.iloc[idx]["review_text"]
            neighbor_rows.append(
                {
                    "model_key": model_key,
                    "rank": rank,
                    "neighbor_idx": int(idx),
                    "cosine_similarity": float(sims[idx]),
                    "sentiment": reviews_df.iloc[idx]["sentiment_label"],
                    "review_text": text_pt,
                }
            )

    neighbor_df = pd.DataFrame(neighbor_rows, columns=cols[:6])
    if neighbor_df.empty:
        return neighbor_df, anchor_sent

    neighbor_df["snippet"] = neighbor_df["review_text"].astype(str).str.slice(0, 140)
    if "review_id" in reviews_df.columns:
        review_id_map = reviews_df["review_id"].astype(str).reset_index(drop=True).to_dict()
        neighbor_df["snippet_en"] = neighbor_df.apply(
            lambda r: to_en_safe(
                r["review_text"],
                review_id=review_id_map.get(int(r["neighbor_idx"])),
            ),
            axis=1,
        ).astype(str).str.slice(0, 140)
    else:
        uniq_text = neighbor_df["review_text"].astype(str).unique().tolist()
        en_map = {t: to_en_safe(t) for t in uniq_text}
        neighbor_df["snippet_en"] = neighbor_df["review_text"].astype(str).map(en_map).str.slice(0, 140)
    neighbor_df["same_sentiment"] = neighbor_df["sentiment"] == anchor_sent
    return neighbor_df, anchor_sent


def summarize_neighbors(neighbor_df: pd.DataFrame, anchor_sent: str) -> pd.DataFrame:
    """Render nearest-neighbor summary table + same-sentiment share chart."""
    from IPython.display import display

    if neighbor_df.empty:
        print("No neighbor rows available. Run embedding extraction first.")
        return pd.DataFrame(columns=["model_key", "share_same_sentiment"])

    cols = [
        "model_key",
        "rank",
        "cosine_similarity",
        "sentiment",
        "same_sentiment",
        "snippet",
        "snippet_en",
    ]
    display(neighbor_df[cols].sort_values(["model_key", "rank"]).reset_index(drop=True))

    summary = (
        neighbor_df.groupby("model_key")["same_sentiment"]
        .mean()
        .reset_index()
        .rename(columns={"same_sentiment": "share_same_sentiment"})
    )
    display(summary)

    fig_bar = px.bar(
        summary,
        x="model_key",
        y="share_same_sentiment",
        text="share_same_sentiment",
        range_y=[0, 1],
        title=f"Share of k-NN with same sentiment as anchor (sentiment = {anchor_sent})",
    )
    fig_bar.show()
    return summary


def plot_local_neighbourhood(
    anchor_idx: int,
    active_matrix: np.ndarray,
    reviews_df: pd.DataFrame,
    model_key: str,
    random_state: int,
    top_k: int = 20,
) -> None:
    """Plot local PCA neighborhood around one anchor review."""
    sims = cosine_similarity(active_matrix[anchor_idx : anchor_idx + 1], active_matrix).ravel()
    nearest_idx = np.argsort(sims)[::-1][1 : top_k + 1]
    indices = np.r_[anchor_idx, nearest_idx]
    X_local = active_matrix[indices]

    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X_local)

    df_local = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "idx": indices,
            "role": ["anchor"] + ["neighbor"] * len(nearest_idx),
            "sentiment": reviews_df.iloc[indices]["sentiment_label"].values,
            "review_text": reviews_df.iloc[indices]["review_text"].values,
        }
    )
    df_local["snippet"] = df_local["review_text"].astype(str).str.slice(0, 140)
    if "review_id" in reviews_df.columns:
        df_local["review_id"] = reviews_df.iloc[indices]["review_id"].astype(str).values
        df_local["snippet_en"] = df_local.apply(
            lambda r: to_en_safe(r["review_text"], review_id=r["review_id"]),
            axis=1,
        ).astype(str).str.slice(0, 140)
    else:
        uniq_text = df_local["review_text"].astype(str).unique().tolist()
        en_map = {t: to_en_safe(t) for t in uniq_text}
        df_local["snippet_en"] = df_local["review_text"].astype(str).map(en_map).str.slice(0, 140)

    fig_scatter = px.scatter(
        df_local,
        x="x",
        y="y",
        color="sentiment",
        symbol="role",
        hover_data=["idx", "snippet", "snippet_en"],
        title=f"k-NN neighbourhood around anchor (model = {model_key})",
    )
    fig_scatter.show()
