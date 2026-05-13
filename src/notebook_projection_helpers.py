"""Projection-map plotting helpers used by the analysis notebook."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans


def ensure_model_labels(
    model_key: str,
    X_model: np.ndarray,
    phase3_by_model: dict,
    random_state: int,
    k_grid: tuple[int, ...],
) -> np.ndarray:
    """Ensure model labels align with embedding rows; recompute fallback if needed."""
    labels_raw = phase3_by_model.get(model_key, {}).get("labels", None)
    n_rows = int(len(X_model))
    if labels_raw is not None:
        labels_arr = np.asarray(labels_raw)
        if labels_arr.shape[0] == n_rows:
            return labels_arr.astype(int)
        print(
            f"[{model_key}] label-size mismatch: labels={labels_arr.shape[0]:,} vs embeddings={n_rows:,}. "
            "Recomputing KMeans labels for visualization."
        )

    best_k_guess = int(phase3_by_model.get(model_key, {}).get("best_k", 0) or 0)
    max_valid_k = max(2, n_rows - 1)
    if best_k_guess < 2 or best_k_guess > max_valid_k:
        grid_default = int(np.median(k_grid)) if len(k_grid) > 0 else 7
        best_k_guess = min(max(2, grid_default), max_valid_k)

    try:
        km_fix = KMeans(n_clusters=best_k_guess, random_state=random_state, n_init=10)
        labels_fix = km_fix.fit_predict(X_model).astype(int)
    except Exception as exc:
        print(f"[{model_key}] fallback KMeans failed ({exc}); assigning a single cluster.")
        labels_fix = np.zeros(n_rows, dtype=int)

    phase3_by_model[model_key]["labels"] = labels_fix
    phase3_by_model[model_key]["best_k"] = int(best_k_guess)
    return labels_fix


def collect_method_frame(
    method_name: str,
    viz_by_model: dict[str, dict[str, pd.DataFrame]],
    model_run_order: list[str],
    max_points: int = 2500,
    random_state: int = 42,
) -> pd.DataFrame:
    """Collect sampled method-level projection rows across models."""
    rng = np.random.default_rng(random_state)
    frames = []
    for model_key in model_run_order:
        if method_name not in viz_by_model.get(model_key, {}):
            continue
        df = viz_by_model[model_key][method_name].copy()
        if len(df) > max_points:
            idx = np.sort(rng.choice(len(df), size=max_points, replace=False))
            df = df.iloc[idx].copy()
        df["model_key"] = model_key
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["snippet"] = out["review_text"].astype(str).str.slice(0, 160)
    return out


def plot_topic_circle_overlay(
    viz_method: pd.DataFrame,
    method_name: str,
    model_run_order: list[str],
    sentiment_palette: dict[str, str],
    sentiment_order: list[str],
    sentiment_markers: dict[str, str],
    topic_order: list[str],
    topic_colors: dict[str, str],
    min_topic_points: int = 20,
    radius_quantile: float = 0.60,
) -> None:
    """Plot topic circles over sentiment-colored projection points."""
    if viz_method.empty or "topic_proxy" not in viz_method.columns:
        return

    model_order = [m for m in model_run_order if m in set(viz_method["model_key"].tolist())]
    if not model_order:
        return

    fig, axes = plt.subplots(1, len(model_order), figsize=(7.2 * len(model_order), 6.4), squeeze=False)
    axes = axes[0]
    topics_seen: list[str] = []

    for ax, model_key in zip(axes, model_order):
        d = viz_method[viz_method["model_key"] == model_key].copy()
        d["topic_proxy"] = d["topic_proxy"].astype(str).str.lower()

        for sent in sentiment_order:
            sd = d[d["sentiment"] == sent]
            if sd.empty:
                continue
            ax.scatter(
                sd["x"].to_numpy(),
                sd["y"].to_numpy(),
                s=26,
                alpha=0.72,
                marker=sentiment_markers[sent],
                c=sentiment_palette[sent],
                linewidths=0.25,
                edgecolors="white",
                zorder=2,
            )

        span = float(max(d["x"].max() - d["x"].min(), d["y"].max() - d["y"].min()))
        for topic in topic_order:
            td = d[d["topic_proxy"] == topic]
            if len(td) < min_topic_points:
                continue

            x = td["x"].to_numpy(dtype=float)
            y = td["y"].to_numpy(dtype=float)
            cx = float(np.mean(x))
            cy = float(np.mean(y))
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            radius = float(np.quantile(dist, radius_quantile))
            radius = min(radius, 0.24 * span)
            if not np.isfinite(radius) or radius <= 0:
                continue

            color = topic_colors[topic]
            ax.add_patch(
                Circle(
                    (cx, cy),
                    radius=radius,
                    fill=True,
                    linewidth=2.0,
                    edgecolor=color,
                    facecolor=color,
                    alpha=0.10,
                    zorder=3,
                )
            )
            ax.text(
                cx,
                cy + radius + 0.02 * max(1.0, float(d["y"].std())),
                topic,
                color=color,
                fontsize=8.5,
                ha="center",
                va="bottom",
                fontweight="bold",
                bbox={"facecolor": "white", "alpha": 0.75, "pad": 1.2, "edgecolor": "none"},
                zorder=4,
            )
            if topic not in topics_seen:
                topics_seen.append(topic)

        ax.set_title(model_key)
        ax.set_xlabel("component 1")
        ax.set_ylabel("component 2")

    sent_handles = [
        Line2D(
            [0],
            [0],
            marker=sentiment_markers[sent],
            color="none",
            markerfacecolor=sentiment_palette[sent],
            markeredgecolor="white",
            markeredgewidth=0.3,
            markersize=9,
            label=f"{sent} (points)",
        )
        for sent in sentiment_order
    ]
    topic_handles = [Line2D([0], [0], color=topic_colors[t], linewidth=2.3, label=f"{t} (circle)") for t in topics_seen]

    fig.legend(
        handles=sent_handles + topic_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.suptitle(f"{method_name}: sentiment projection with topic-region circles", y=1.01)
    fig.tight_layout(rect=[0.0, 0.10, 1.0, 0.98])
    plt.show()


def plot_method_panels(
    viz_method: pd.DataFrame,
    method_name: str,
    model_run_order: list[str],
    sentiment_palette: dict[str, str],
    sentiment_order: list[str],
    sentiment_markers: dict[str, str],
    topic_order: list[str],
    topic_colors: dict[str, str],
) -> None:
    """Render projection panels for cluster/sentiment/overlay views."""
    model_order = [m for m in model_run_order if m in set(viz_method["model_key"].tolist())]
    cluster_vals = viz_method["cluster"].astype(str).unique().tolist()
    cluster_order = sorted(cluster_vals, key=lambda x: (0, int(x)) if str(x).lstrip("-").isdigit() else (1, x))

    fig_cluster = px.scatter(
        viz_method,
        x="x",
        y="y",
        color="cluster",
        facet_col="model_key",
        category_orders={"model_key": model_order, "cluster": cluster_order},
        hover_data={"review_id": True, "sentiment": True, "snippet": True},
        title=f"{method_name}: cluster geometry by model",
        height=560,
    )
    fig_cluster.for_each_annotation(lambda a: a.update(text=a.text.replace("model_key=", "")))
    fig_cluster.show()

    fig_sent = px.scatter(
        viz_method,
        x="x",
        y="y",
        color="sentiment",
        symbol="sentiment",
        color_discrete_map=sentiment_palette,
        category_orders={"sentiment": sentiment_order, "model_key": model_order},
        facet_col="model_key",
        hover_data={"review_id": True, "cluster": True, "snippet": True},
        title=f"{method_name}: sentiment layer by model",
        height=560,
    )
    fig_sent.for_each_annotation(lambda a: a.update(text=a.text.replace("model_key=", "")))
    fig_sent.show()

    fig_overlay = px.scatter(
        viz_method,
        x="x",
        y="y",
        color="sentiment",
        symbol="cluster",
        color_discrete_map=sentiment_palette,
        category_orders={"sentiment": sentiment_order, "model_key": model_order, "cluster": cluster_order},
        opacity=0.72,
        facet_col="model_key",
        hover_data={"review_id": True, "cluster": True, "snippet": True},
        title=f"{method_name}: overlay A (sentiment color + cluster marker)",
        height=620,
    )
    fig_overlay.for_each_annotation(lambda a: a.update(text=a.text.replace("model_key=", "")))
    fig_overlay.show()

    fig_overlay_inverse = px.scatter(
        viz_method,
        x="x",
        y="y",
        color="cluster",
        symbol="sentiment",
        category_orders={"sentiment": sentiment_order, "model_key": model_order, "cluster": cluster_order},
        opacity=0.72,
        facet_col="model_key",
        hover_data={"review_id": True, "cluster": True, "snippet": True},
        title=f"{method_name}: overlay B (cluster color + sentiment symbol)",
        height=620,
    )
    fig_overlay_inverse.for_each_annotation(lambda a: a.update(text=a.text.replace("model_key=", "")))
    fig_overlay_inverse.show()

    plot_topic_circle_overlay(
        viz_method=viz_method,
        method_name=method_name,
        model_run_order=model_run_order,
        sentiment_palette=sentiment_palette,
        sentiment_order=sentiment_order,
        sentiment_markers=sentiment_markers,
        topic_order=topic_order,
        topic_colors=topic_colors,
    )

