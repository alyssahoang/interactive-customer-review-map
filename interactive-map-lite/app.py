from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_TITLE = "Customer Review Explorer"
APP_SUBTITLE = "Explore review themes, mood, and example comments for Olist in a business-friendly map view."

st.set_page_config(page_title=APP_TITLE, page_icon="map", layout="wide")

APP_DIR = Path(__file__).resolve().parent


def _resolve_project_root() -> Path:
    """Resolve package root robustly for both standalone and monorepo usage."""
    # Preferred layout: <project_root>/interactive-map-lite/app.py and <project_root>/data
    direct_parent = APP_DIR.parent
    if (direct_parent / "data").exists():
        return direct_parent

    # Monorepo fallback: <repo_root>/submission/interactive-map-lite/app.py
    candidate = direct_parent.parent / "submission"
    if (candidate / "data").exists():
        return candidate

    # Last resort: previous behavior.
    return direct_parent


PROJECT_ROOT = _resolve_project_root()
ANALYSIS_V0_PHASE3_ROOT = PROJECT_ROOT / "data" 
ANALYSIS_V0_PROJECTION_ROOT = ANALYSIS_V0_PHASE3_ROOT / "projection_views"
ANALYSIS_V0_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "analysis-v1"
OLIST_PT_REVIEWS_PATH = PROJECT_ROOT / "data" / "olist_order_reviews_dataset.csv"
OLIST_TRANSLATED_REVIEWS_PATH = PROJECT_ROOT / "data" / "olist_order_reviews_dataset_translated.csv"
SENTIMENT_COLOR_MAP = {
    "negative": "#e4572e",
    "neutral": "#8e8e8e",
    "positive": "#2a9df4",
    "other": "#b084cc",
}
SENTIMENT_DEPTH_MAP = {
    "negative": -1.0,
    "neutral": 0.0,
    "positive": 1.0,
    "other": 0.0,
}
MODEL_LABELS = {
    "bertimbau": "BERTimbau",
    "multilingual_minilm": "MiniLM",
    "openclip_text": "OpenCLIP",
}
PROJECTION_LABELS = {
    "umap": "UMAP",
    "tsne": "tSNE",
    "pca": "PCA",
}
PROJECTION_ORDER = ["UMAP", "tSNE", "PCA"]
TOPIC_REGION_COLOR_MAP = {
    "delivery": "#1f77b4",
    "product": "#2ca02c",
    "support": "#ff7f0e",
    "refund": "#d62728",
    "other": "#7f7f7f",
    "price": "#9467bd",
}
TOPIC_REGION_FALLBACK_COLORS = [
    "#17becf",
    "#8c564b",
    "#e377c2",
    "#bcbd22",
    "#7f7f7f",
    "#1f77b4",
    "#ff9896",
    "#98df8a",
]
CLUSTER_REGION_COLORS = px.colors.qualitative.Set2
DEFAULT_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "was",
    "were",
    "are",
    "you",
    "your",
    "have",
    "has",
    "had",
    "not",
    "but",
    "very",
    "from",
    "they",
    "them",
    "our",
    "just",
    "all",
    "too",
    "it",
    "its",
    "a",
    "an",
    "to",
    "of",
    "in",
    "on",
    "is",
    "be",
    "as",
    "at",
    "my",
    "me",
    "we",
    "us",
    "de",
    "do",
    "da",
    "das",
    "dos",
    "e",
    "em",
    "no",
    "na",
    "foi",
    "com",
    "para",
    "por",
    "muito",
    "mais",
    "produto",
    "order",
    "delivery",
    "product",
}


TOPIC_PHRASE_EN_MAP = {
    "nÃ£o recebi": "not received",
    "do prazo": "on time",
    "antes do": "ahead of schedule",
}

TOPIC_TOKEN_EN_MAP = {
    "nÃ£o": "not",
    "recebi": "received",
    "produto": "product",
    "ainda": "yet",
    "de": "of",
    "veio": "arrived",
    "que": "that",
    "muito": "very",
    "recomendo": "recommend",
    "gostei": "liked",
    "tudo": "everything",
    "bom": "good",
    "prazo": "delivery time",
    "do": "the",
    "antes": "before",
    "qualidade": "quality",
    "Ã³timo": "great",
    "entrega": "delivery",
    "loja": "store",
    "compra": "purchase",
    "comprar": "buy",
}


def _topic_keyword_to_english(topic_keyword: str) -> str:
    if topic_keyword is None:
        return ""
    text = str(topic_keyword).strip().lower()
    if not text:
        return ""

    parts = [p.strip() for p in text.split("|")]
    out_parts: list[str] = []
    for part in parts:
        if not part:
            continue
        if part in TOPIC_PHRASE_EN_MAP:
            out_parts.append(TOPIC_PHRASE_EN_MAP[part])
            continue

        tokens = part.split()
        translated_tokens = [TOPIC_TOKEN_EN_MAP.get(tok, tok) for tok in tokens]
        out_parts.append(" ".join(translated_tokens))

    return " | ".join(out_parts)


def _render_stakeholder_intro() -> None:
    st.markdown("#### How To Use This App")
    st.markdown(
        "Start with `Business Summary` for key issues and strengths, then open `Projection Map` to inspect review examples."
    )
    st.markdown(
        "Scope is fixed to Olist; use sidebar filters for date, mood, topic, and review group."
    )


def _render_sidebar_glossary() -> None:
    with st.sidebar.expander("Quick guide & glossary", expanded=False):
        st.markdown("1. Start in `Business Summary`.")
        st.markdown("2. Filter by period, topic, mood, and review group.")
        st.markdown("3. Open `Projection Map`: each dot is one review.")
        st.markdown("4. Nearby dots indicate similar review meaning.")
        st.markdown("5. Region circles are optional visual guides, not hard boundaries.")

        glossary_rows = [
            {
                "Term": "Projection (UMAP/tSNE/PCA)",
                "Plain meaning": "How embeddings are arranged into a 2D map.",
            },
            {
                "Term": "Review group (cluster)",
                "Plain meaning": "Reviews with similar semantics grouped together.",
            },
            {
                "Term": "Topic proxy",
                "Plain meaning": "Rule-based topic label from review text.",
            },
            {
                "Term": "Mood (sentiment)",
                "Plain meaning": "Predicted tone: negative, neutral, or positive.",
            },
            {
                "Term": "Best k",
                "Plain meaning": "Selected number of clusters in KMeans diagnostics.",
            },
            {
                "Term": "Silhouette",
                "Plain meaning": "Separation quality score; higher usually means cleaner separation.",
            },
            {
                "Term": "Purity",
                "Plain meaning": "How consistent cluster members are under a label (topic or mood).",
            },
            {
                "Term": "kNN",
                "Plain meaning": "Nearest-neighbor retrieval for similar reviews on the map.",
            },
        ]
        st.dataframe(pd.DataFrame(glossary_rows), width="stretch", hide_index=True)


def _candidate_artifact_roots() -> list[Path]:
    # Strict source: artifacts exported by notebook/olist_negative_sentiment_geometry_analysis.ipynb
    return [ANALYSIS_V0_ARTIFACT_ROOT]


def _candidate_projection_roots() -> list[Path]:
    # Strict analysis-v0 source for map coordinates.
    return [ANALYSIS_V0_PROJECTION_ROOT]


def _candidate_topic_proxy_pack_paths() -> list[Path]:
    # Optional analysis-v0 export (if present). Primary fallback is label_index.parquet per model.
    return [
        ANALYSIS_V0_PROJECTION_ROOT / "topic_proxy_lookup.csv",
        ANALYSIS_V0_PROJECTION_ROOT / "topic_proxy_lookup.parquet",
    ]


def _candidate_phase3_roots() -> list[Path]:
    # Strict analysis-v0 source for Phase 3 diagnostics.
    return [ANALYSIS_V0_PHASE3_ROOT]


def _candidate_text_lookup_paths() -> list[Path]:
    # Optional analysis-v0 export for richer snippets/dates.
    return [ANALYSIS_V0_PROJECTION_ROOT / "review_text_lookup.csv"]


def _candidate_wordcloud_pack_paths() -> list[Path]:
    # Optional analysis-v0 export for precomputed n-grams.
    return [
        ANALYSIS_V0_PROJECTION_ROOT / "wordcloud_terms.csv",
        ANALYSIS_V0_PROJECTION_ROOT / "wordcloud_terms.parquet",
    ]


def _discover_runs() -> tuple[Path | None, list[Path]]:
    for root in _candidate_artifact_roots():
        if not root.exists():
            continue
        runs = sorted(
            [
                p
                for p in root.iterdir()
                if p.is_dir() and (p / "semantic_map_points.csv").exists()
            ],
            key=lambda p: p.name,
        )
        if runs:
            return root, runs
    return None, []


def _discover_projection_files() -> tuple[Path | None, dict[str, Path]]:
    for root in _candidate_projection_roots():
        if not root.exists():
            continue

        found = sorted(
            [
                p
                for p in root.iterdir()
                if p.is_file() and p.name.startswith("viz_") and p.suffix.lower() in {".csv", ".parquet"}
            ],
            key=lambda p: p.name,
        )
        if not found:
            continue

        label_to_path: dict[str, Path] = {}
        for path in found:
            method_key = path.stem.replace("viz_", "", 1).lower()
            label = PROJECTION_LABELS.get(method_key, method_key.upper())
            prev = label_to_path.get(label)
            # Prefer CSV to avoid optional parquet dependency on Streamlit Cloud.
            if prev is None or (prev.suffix.lower() == ".parquet" and path.suffix.lower() == ".csv"):
                label_to_path[label] = path

        if not label_to_path:
            continue

        ordered: dict[str, Path] = {}
        for label in PROJECTION_ORDER:
            if label in label_to_path:
                ordered[label] = label_to_path[label]
        for label in sorted(label_to_path.keys()):
            if label not in ordered:
                ordered[label] = label_to_path[label]

        return root, ordered

    return None, {}


@st.cache_data(show_spinner=False)
def _read_csv(path_str: str, mtime_ns: int) -> pd.DataFrame:
    return pd.read_csv(path_str)


@st.cache_data(show_spinner=False)
def _read_parquet(path_str: str, mtime_ns: int) -> pd.DataFrame:
    return pd.read_parquet(path_str)


def _load_phase3_table(filename: str) -> pd.DataFrame:
    for root in _candidate_phase3_roots():
        path = root / filename
        df = _load_csv(path)
        if not df.empty:
            return df
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _load_phase3_model_comparison() -> pd.DataFrame:
    df = _load_phase3_table("phase3_model_comparison.csv")
    if df.empty:
        return pd.DataFrame()

    out = df.copy()
    if "model_key" in out.columns:
        out["model_key"] = out["model_key"].fillna("unknown").astype(str).str.strip().str.lower()
    if "best_k" in out.columns:
        out["best_k"] = pd.to_numeric(out["best_k"], errors="coerce")

    numeric_cols = [
        "kmeans_silhouette",
        "kmeans_purity_topic_proxy",
        "kmeans_purity_sentiment",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


@st.cache_data(show_spinner=False)
def _load_projection_quality_table() -> pd.DataFrame:
    df = _load_phase3_table("projection_quality_all_models.csv")
    if df.empty:
        return pd.DataFrame()

    out = df.copy()
    if "model_key" in out.columns:
        out["model_key"] = out["model_key"].fillna("unknown").astype(str).str.strip().str.lower()
    if "projection" in out.columns:
        out["projection"] = out["projection"].fillna("unknown").astype(str).str.strip().str.upper()

    numeric_cols = [
        "silhouette_sentiment",
        "silhouette_topic_proxy",
        "delta_topic_minus_sentiment",
        "n_rows",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return _read_csv(str(path), int(path.stat().st_mtime_ns))
    except Exception:
        return pd.DataFrame()


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return _read_parquet(str(path), int(path.stat().st_mtime_ns))
    except Exception:
        return pd.DataFrame()


def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_csv(path)
    if suffix == ".parquet":
        return _load_parquet(path)
    return pd.DataFrame()


def _normalize_map_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    rename_map = {
        "cluster_id": "cluster",
        "text": "review_text",
        "snippet_text": "snippet",
        "review_comment_message": "review_text",
    }
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    required = {"x", "y", "cluster", "sentiment"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    df = df.copy()

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x", "y"]).reset_index(drop=True)

    if "review_id" not in df.columns:
        df["review_id"] = df.index.astype(str)

    if "model_key" not in df.columns:
        df["model_key"] = "unknown"

    df["model_key"] = df["model_key"].fillna("unknown").astype(str).str.strip().str.lower()
    df["model_label"] = df["model_key"].map(MODEL_LABELS).fillna(df["model_key"])

    if "dataset_key" not in df.columns:
        df["dataset_key"] = "all"
    df["dataset_key"] = df["dataset_key"].fillna("all").astype(str).str.strip().str.lower()
    df.loc[df["dataset_key"].eq(""), "dataset_key"] = "all"

    df["sentiment"] = df["sentiment"].fillna("unknown").astype(str).str.strip().str.lower()
    known = {"negative", "neutral", "positive"}
    df.loc[~df["sentiment"].isin(known), "sentiment"] = "other"

    df["cluster"] = df["cluster"].fillna("unknown").astype(str)
    df["cluster_label"] = "cluster_" + df["cluster"]

    if "topic_proxy" not in df.columns:
        df["topic_proxy"] = "unknown"
    df["topic_proxy"] = df["topic_proxy"].fillna("unknown").astype(str).str.strip().str.lower()
    df.loc[df["topic_proxy"].eq(""), "topic_proxy"] = "unknown"

    if "snippet_en" not in df.columns:
        if "snippet" in df.columns:
            df["snippet_en"] = df["snippet"].fillna("").astype(str)
        elif "review_text_en" in df.columns:
            df["snippet_en"] = df["review_text_en"].fillna("").astype(str)
        elif "review_text" in df.columns:
            df["snippet_en"] = df["review_text"].fillna("").astype(str)
        else:
            df["snippet_en"] = ""

    if "snippet_pt" not in df.columns:
        if "review_text_pt" in df.columns:
            df["snippet_pt"] = df["review_text_pt"].fillna("").astype(str)
        elif "snippet" in df.columns:
            df["snippet_pt"] = df["snippet"].fillna("").astype(str)
        elif "review_text" in df.columns:
            df["snippet_pt"] = df["review_text"].fillna("").astype(str)
        else:
            df["snippet_pt"] = ""

    if "review_text" in df.columns:
        review_text = df["review_text"].fillna("").astype(str)
        english_only_mask = df["dataset_key"].isin({"womens_ecommerce", "women_ecommerce"})
        fill_en_mask = english_only_mask & df["snippet_en"].eq("")
        df.loc[fill_en_mask, "snippet_en"] = review_text[fill_en_mask]
        clear_pt_mask = english_only_mask & df["snippet_pt"].eq(review_text)
        df.loc[clear_pt_mask, "snippet_pt"] = ""

    df["snippet_en"] = df["snippet_en"].fillna("").astype(str).str.slice(0, 220)
    df["snippet_pt"] = df["snippet_pt"].fillna("").astype(str).str.slice(0, 220)

    if "review_creation_date" not in df.columns:
        df["review_creation_date"] = ""
    df["review_creation_date"] = df["review_creation_date"].fillna("").astype(str)
    df["review_date"] = pd.to_datetime(df["review_creation_date"], errors="coerce")

    return df


def _cluster_sort_key(label: str) -> tuple[int, str]:
    suffix = label.replace("cluster_", "", 1)
    if suffix.isdigit():
        return int(suffix), label
    return 10**9, label


def _cluster_numeric_series(cluster_labels: pd.Series) -> pd.Series:
    suffix = cluster_labels.astype(str).str.replace("cluster_", "", regex=False)
    as_num = pd.to_numeric(suffix, errors="coerce")
    if as_num.notna().all():
        return as_num.astype(float)
    codes, _ = pd.factorize(cluster_labels.astype(str))
    return pd.Series(codes, index=cluster_labels.index, dtype=float)


def _topic_numeric_series(topic_proxy: pd.Series) -> pd.Series:
    topic_str = topic_proxy.fillna("unknown").astype(str)
    ordered = sorted(topic_str.unique().tolist())
    mapping = {name: float(idx) for idx, name in enumerate(ordered)}
    return topic_str.map(mapping).astype(float)


def _deterministic_jitter(seed_values: pd.Series, scale: float = 0.2) -> pd.Series:
    hashes = pd.util.hash_pandas_object(seed_values.astype(str), index=False).astype("uint64")
    return ((hashes % 1000) / 1000.0 - 0.5) * (2.0 * scale)


def _derive_depth(df: pd.DataFrame, depth_mode: str) -> pd.Series:
    if depth_mode in {"Cluster index", "Review group"}:
        base = _cluster_numeric_series(df["cluster_label"])
    elif depth_mode in {"Topic proxy code", "Topic"}:
        base = _topic_numeric_series(df["topic_proxy"])
    else:
        base = df["sentiment"].map(SENTIMENT_DEPTH_MAP).fillna(0.0).astype(float)

    return (base + _deterministic_jitter(df["review_id"], scale=0.2)).astype(float)


@st.cache_data(show_spinner=False)
def _load_topic_proxy_lookup() -> pd.DataFrame:
    for packed_path in _candidate_topic_proxy_pack_paths():
        if packed_path.exists():
            df = _load_table(packed_path)
            needed = {"model_key", "review_id", "topic_proxy"}
            if needed.issubset(df.columns):
                out = df[["model_key", "review_id", "topic_proxy"]].copy()
                out["model_key"] = out["model_key"].astype(str).str.lower()
                out["review_id"] = out["review_id"].astype(str)
                out["topic_proxy"] = out["topic_proxy"].fillna("unknown").astype(str).str.lower()
                return out

    frames: list[pd.DataFrame] = []
    for phase3_root in _candidate_phase3_roots():
        if not phase3_root.exists():
            continue

        loaded_any = False
        for model_key in MODEL_LABELS.keys():
            label_path = phase3_root / model_key / "label_index.parquet"
            if not label_path.exists():
                continue

            label_df = _load_parquet(label_path)
            if label_df.empty or not {"review_id", "topic_proxy"}.issubset(label_df.columns):
                continue

            slim = label_df[["review_id", "topic_proxy"]].copy()
            slim["model_key"] = model_key
            slim["review_id"] = slim["review_id"].astype(str)
            slim["topic_proxy"] = slim["topic_proxy"].fillna("unknown").astype(str).str.lower()
            frames.append(slim)
            loaded_any = True

        if loaded_any:
            break

    if not frames:
        return pd.DataFrame(columns=["model_key", "review_id", "topic_proxy"])

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["model_key", "review_id"], keep="first")
    return out


@st.cache_data(show_spinner=False)
def _load_translated_review_lookup() -> pd.DataFrame:
    path = OLIST_TRANSLATED_REVIEWS_PATH
    if not path.exists():
        return pd.DataFrame(columns=["review_id", "snippet_en_translated", "review_creation_date_translated"])

    df = _load_csv(path)
    if df.empty or "review_id" not in df.columns:
        return pd.DataFrame(columns=["review_id", "snippet_en_translated", "review_creation_date_translated"])

    out = pd.DataFrame({"review_id": df["review_id"].astype(str)})
    if "review_comment_message" in df.columns:
        out["snippet_en_translated"] = df["review_comment_message"].fillna("").astype(str).str.slice(0, 220)
    else:
        out["snippet_en_translated"] = ""

    if "review_creation_date" in df.columns:
        out["review_creation_date_translated"] = pd.to_datetime(df["review_creation_date"], errors="coerce")
    else:
        out["review_creation_date_translated"] = pd.NaT

    out = out.drop_duplicates(subset=["review_id"], keep="first")
    return out


@st.cache_data(show_spinner=False)
def _load_olist_pt_review_lookup() -> pd.DataFrame:
    path = OLIST_PT_REVIEWS_PATH
    if not path.exists():
        return pd.DataFrame(columns=["review_id", "snippet_pt_raw", "review_creation_date_raw"])

    df = _load_csv(path)
    if df.empty or "review_id" not in df.columns:
        return pd.DataFrame(columns=["review_id", "snippet_pt_raw", "review_creation_date_raw"])

    out = pd.DataFrame({"review_id": df["review_id"].astype(str)})
    if "review_comment_message" in df.columns:
        out["snippet_pt_raw"] = df["review_comment_message"].fillna("").astype(str).str.slice(0, 220)
    else:
        out["snippet_pt_raw"] = ""

    if "review_creation_date" in df.columns:
        out["review_creation_date_raw"] = pd.to_datetime(df["review_creation_date"], errors="coerce")
    else:
        out["review_creation_date_raw"] = pd.NaT

    out = out.drop_duplicates(subset=["review_id"], keep="first")
    return out


@st.cache_data(show_spinner=False)
def _load_text_lookup() -> pd.DataFrame:
    base_lookup = pd.DataFrame(
        columns=["review_id", "snippet_pt_lookup", "snippet_en_lookup", "review_creation_date_lookup"]
    )

    for path in _candidate_text_lookup_paths():
        if not path.exists():
            continue
        df = _load_csv(path)
        if df.empty or "review_id" not in df.columns:
            continue

        out = pd.DataFrame({"review_id": df["review_id"].astype(str)})
        if "snippet_pt" in df.columns:
            out["snippet_pt_lookup"] = df["snippet_pt"].fillna("").astype(str)
        elif "review_text_pt" in df.columns:
            out["snippet_pt_lookup"] = df["review_text_pt"].fillna("").astype(str)
        else:
            out["snippet_pt_lookup"] = ""

        if "snippet_en" in df.columns:
            out["snippet_en_lookup"] = df["snippet_en"].fillna("").astype(str)
        elif "review_text_en" in df.columns:
            out["snippet_en_lookup"] = df["review_text_en"].fillna("").astype(str)
        else:
            out["snippet_en_lookup"] = ""

        if "review_creation_date" in df.columns:
            out["review_creation_date_lookup"] = pd.to_datetime(df["review_creation_date"], errors="coerce")
        else:
            out["review_creation_date_lookup"] = pd.NaT

        out["snippet_pt_lookup"] = out["snippet_pt_lookup"].str.slice(0, 220)
        out["snippet_en_lookup"] = out["snippet_en_lookup"].str.slice(0, 220)
        base_lookup = out.drop_duplicates(subset=["review_id"], keep="first")
        break

    pt_lookup = _load_olist_pt_review_lookup()
    if base_lookup.empty and not pt_lookup.empty:
        base_lookup = pd.DataFrame({"review_id": pt_lookup["review_id"].astype(str)})
        base_lookup["snippet_pt_lookup"] = pt_lookup["snippet_pt_raw"].fillna("").astype(str)
        base_lookup["snippet_en_lookup"] = ""
        base_lookup["review_creation_date_lookup"] = pd.to_datetime(
            pt_lookup["review_creation_date_raw"], errors="coerce"
        )
    elif not base_lookup.empty and not pt_lookup.empty:
        merged_pt = base_lookup.merge(pt_lookup, on="review_id", how="left")
        pt_missing = merged_pt["snippet_pt_lookup"].fillna("").astype(str).str.strip().eq("")
        merged_pt.loc[pt_missing, "snippet_pt_lookup"] = merged_pt.loc[pt_missing, "snippet_pt_raw"].fillna("")

        base_dates = pd.to_datetime(merged_pt["review_creation_date_lookup"], errors="coerce")
        raw_dates = pd.to_datetime(merged_pt["review_creation_date_raw"], errors="coerce")
        merged_pt["review_creation_date_lookup"] = base_dates.combine_first(raw_dates)

        merged_pt = merged_pt.drop(columns=[c for c in ["snippet_pt_raw", "review_creation_date_raw"] if c in merged_pt.columns])
        base_lookup = merged_pt

    translated_lookup = _load_translated_review_lookup()
    if translated_lookup.empty:
        return base_lookup.drop_duplicates(subset=["review_id"], keep="first")

    if base_lookup.empty:
        merged = pd.DataFrame({"review_id": translated_lookup["review_id"].astype(str)})
        merged["snippet_pt_lookup"] = ""
        merged["snippet_en_lookup"] = translated_lookup["snippet_en_translated"].fillna("").astype(str)
        merged["review_creation_date_lookup"] = pd.to_datetime(
            translated_lookup["review_creation_date_translated"], errors="coerce"
        )
        return merged.drop_duplicates(subset=["review_id"], keep="first")

    merged = base_lookup.merge(translated_lookup, on="review_id", how="left")

    needs_english = (
        merged["snippet_en_lookup"].fillna("").astype(str).str.strip().eq("")
        | (
            merged["snippet_pt_lookup"].fillna("").astype(str).str.strip()
            == merged["snippet_en_lookup"].fillna("").astype(str).str.strip()
        )
    )
    merged.loc[needs_english, "snippet_en_lookup"] = merged.loc[needs_english, "snippet_en_translated"].fillna("")

    if "review_creation_date_lookup" in merged.columns:
        merged["review_creation_date_lookup"] = pd.to_datetime(merged["review_creation_date_lookup"], errors="coerce")
    else:
        merged["review_creation_date_lookup"] = pd.NaT

    if "review_creation_date_translated" in merged.columns:
        trans_dates = pd.to_datetime(merged["review_creation_date_translated"], errors="coerce")
        merged["review_creation_date_lookup"] = merged["review_creation_date_lookup"].combine_first(trans_dates)

    drop_cols = [c for c in ["snippet_en_translated", "review_creation_date_translated"] if c in merged.columns]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)

    merged["snippet_pt_lookup"] = merged["snippet_pt_lookup"].fillna("").astype(str).str.slice(0, 220)
    merged["snippet_en_lookup"] = merged["snippet_en_lookup"].fillna("").astype(str).str.slice(0, 220)
    return merged.drop_duplicates(subset=["review_id"], keep="first")


@st.cache_data(show_spinner=False)
def _load_precomputed_wordcloud_terms() -> pd.DataFrame:
    required = {"model_key", "topic_proxy", "sentiment", "ngram_n", "phrase", "count"}
    for path in _candidate_wordcloud_pack_paths():
        if not path.exists():
            continue
        df = _load_table(path)
        if df.empty or not required.issubset(df.columns):
            continue

        out_cols = ["model_key", "topic_proxy", "sentiment", "ngram_n", "phrase", "count"]
        if "dataset_key" in df.columns:
            out_cols = ["dataset_key"] + out_cols
        out = df[out_cols].copy()
        if "dataset_key" not in out.columns:
            out["dataset_key"] = "all"

        out["dataset_key"] = out["dataset_key"].fillna("all").astype(str).str.strip().str.lower()
        out.loc[out["dataset_key"].eq(""), "dataset_key"] = "all"
        out["model_key"] = out["model_key"].fillna("unknown").astype(str).str.strip().str.lower()
        out["topic_proxy"] = out["topic_proxy"].fillna("unknown").astype(str).str.strip().str.lower()
        out["sentiment"] = out["sentiment"].fillna("all").astype(str).str.strip().str.lower()
        out["ngram_n"] = pd.to_numeric(out["ngram_n"], errors="coerce").fillna(0).astype(int)
        out["phrase"] = out["phrase"].fillna("").astype(str)
        out["count"] = pd.to_numeric(out["count"], errors="coerce").fillna(0).astype(int)
        out = out[(out["ngram_n"] > 0) & out["phrase"].ne("") & (out["count"] > 0)].copy()
        if out.empty:
            continue
        return out

    return pd.DataFrame(columns=["dataset_key", "model_key", "topic_proxy", "sentiment", "ngram_n", "phrase", "count"])


def _attach_topic_proxy(df: pd.DataFrame) -> pd.DataFrame:
    lookup = _load_topic_proxy_lookup()
    if lookup.empty:
        return df

    merged = df.merge(
        lookup,
        on=["model_key", "review_id"],
        how="left",
        suffixes=("", "_lookup"),
    )

    if "topic_proxy_lookup" in merged.columns:
        missing = merged["topic_proxy"].eq("unknown") | merged["topic_proxy"].eq("")
        merged.loc[missing, "topic_proxy"] = merged.loc[missing, "topic_proxy_lookup"].fillna("unknown")
        merged = merged.drop(columns=["topic_proxy_lookup"])

    merged["topic_proxy"] = merged["topic_proxy"].fillna("unknown").astype(str).str.strip().str.lower()
    merged.loc[merged["topic_proxy"].eq(""), "topic_proxy"] = "unknown"
    return merged


def _attach_text_lookup(df: pd.DataFrame) -> pd.DataFrame:
    lookup = _load_text_lookup()
    if lookup.empty:
        out = df.copy()
        if "dataset_key" in out.columns:
            dataset_key = out["dataset_key"].fillna("").astype(str).str.strip().str.lower()
            olist_mask = dataset_key.eq("olist")
            fill_mask = olist_mask & out["snippet_en"].eq("")
            out.loc[fill_mask, "snippet_en"] = out.loc[fill_mask, "snippet_pt"]
        else:
            out.loc[out["snippet_en"].eq(""), "snippet_en"] = out.loc[out["snippet_en"].eq(""), "snippet_pt"]
        if "review_creation_date" not in out.columns:
            out["review_creation_date"] = ""
        out["review_creation_date"] = out["review_creation_date"].fillna("").astype(str)
        out["review_date"] = pd.to_datetime(out["review_creation_date"], errors="coerce")
        return out

    merged = df.merge(lookup, on="review_id", how="left")

    if "snippet_pt_lookup" in merged.columns:
        mask_pt = merged["snippet_pt"].eq("")
        merged.loc[mask_pt, "snippet_pt"] = merged.loc[mask_pt, "snippet_pt_lookup"].fillna("")

    if "snippet" in merged.columns:
        mask_pt = merged["snippet_pt"].fillna("").astype(str).str.strip().eq("")
        merged.loc[mask_pt, "snippet_pt"] = merged.loc[mask_pt, "snippet"].fillna("")
    if "review_text" in merged.columns:
        mask_pt = merged["snippet_pt"].fillna("").astype(str).str.strip().eq("")
        merged.loc[mask_pt, "snippet_pt"] = merged.loc[mask_pt, "review_text"].fillna("")

    if "snippet_en_lookup" in merged.columns:
        translated_available = merged["snippet_en_lookup"].fillna("").astype(str).str.strip().ne("")
        merged.loc[translated_available, "snippet_en"] = merged.loc[translated_available, "snippet_en_lookup"].fillna("")
        mask_en = merged["snippet_en"].eq("")
        merged.loc[mask_en, "snippet_en"] = merged.loc[mask_en, "snippet_en_lookup"].fillna("")

    merged["snippet_pt"] = merged["snippet_pt"].fillna("").astype(str).str.slice(0, 220)
    merged["snippet_en"] = merged["snippet_en"].fillna("").astype(str).str.slice(0, 220)

    if "dataset_key" in merged.columns:
        dataset_key = merged["dataset_key"].fillna("").astype(str).str.strip().str.lower()
        olist_mask = dataset_key.eq("olist")
        mask_en_empty = merged["snippet_en"].eq("")
        fill_mask = olist_mask & mask_en_empty
        merged.loc[fill_mask, "snippet_en"] = merged.loc[fill_mask, "snippet_pt"]
    else:
        mask_en_empty = merged["snippet_en"].eq("")
        merged.loc[mask_en_empty, "snippet_en"] = merged.loc[mask_en_empty, "snippet_pt"]

    if "review_date" not in merged.columns:
        base_dates = pd.to_datetime(merged.get("review_creation_date", pd.Series(index=merged.index)), errors="coerce")
        merged["review_date"] = base_dates
    if "review_creation_date_lookup" in merged.columns:
        lookup_dates = pd.to_datetime(merged["review_creation_date_lookup"], errors="coerce")
        merged["review_date"] = pd.to_datetime(merged["review_date"], errors="coerce").combine_first(lookup_dates)

    base_creation = merged.get("review_creation_date", pd.Series("", index=merged.index)).fillna("").astype(str)
    derived_creation = pd.to_datetime(merged["review_date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    merged["review_creation_date"] = derived_creation.where(derived_creation.ne(""), base_creation)

    drop_cols = [
        c
        for c in ["snippet_pt_lookup", "snippet_en_lookup", "review_creation_date_lookup"]
        if c in merged.columns
    ]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)

    return merged


def _load_projection_points(projection_path: Path) -> pd.DataFrame:
    df = _load_table(projection_path)
    if df.empty:
        return pd.DataFrame()
    df = _normalize_map_df(df)
    if df.empty:
        return pd.DataFrame()
    df = _attach_topic_proxy(df)
    df = _attach_text_lookup(df)
    return df


def _load_artifact_points(run_dir: Path, topic_df: pd.DataFrame) -> pd.DataFrame:
    map_df = _normalize_map_df(_load_csv(run_dir / "semantic_map_points.csv"))
    if map_df.empty:
        return pd.DataFrame()

    if not topic_df.empty and {"cluster_id", "topic_name_keyword"}.issubset(topic_df.columns):
        cluster_topic = topic_df[["cluster_id", "topic_name_keyword"]].copy()
        cluster_topic["cluster_label"] = "cluster_" + cluster_topic["cluster_id"].astype(str)
        cluster_topic = cluster_topic.drop_duplicates(subset=["cluster_label"], keep="first")
        map_df = map_df.merge(cluster_topic[["cluster_label", "topic_name_keyword"]], on="cluster_label", how="left")
        map_df["topic_proxy"] = map_df["topic_proxy"].where(
            ~map_df["topic_proxy"].eq("unknown"),
            map_df["topic_name_keyword"].fillna("unknown").astype(str).str.lower(),
        )
        map_df = map_df.drop(columns=["topic_name_keyword"])

    map_df = _attach_text_lookup(map_df)
    return map_df


def _default_model_from_run_summary(run_summary: dict) -> str:
    cfg = run_summary.get("config", {}) if isinstance(run_summary, dict) else {}
    model_key = str(cfg.get("model_key", "")).strip().lower()
    if model_key in MODEL_LABELS:
        return model_key
    return "multilingual_minilm"


def _load_topic_table(run_dir: Path) -> pd.DataFrame:
    for filename in ["topic_table.csv", "topic_table_all_models.csv"]:
        df = _load_csv(run_dir / filename)
        if not df.empty:
            return df
    return pd.DataFrame()


def _load_dataset_lookup(run_dir: Path | None) -> pd.DataFrame:
    if run_dir is None:
        _, runs = _discover_runs()
        if runs:
            run_dir = runs[-1]
        else:
            return pd.DataFrame(columns=["review_id", "dataset_key"])

    semantic_path = run_dir / "semantic_map_points.csv"
    semantic_df = _load_csv(semantic_path)
    if semantic_df.empty or not {"review_id", "dataset_key"}.issubset(semantic_df.columns):
        return pd.DataFrame(columns=["review_id", "dataset_key"])

    out = semantic_df[["review_id", "dataset_key"]].copy()
    out["review_id"] = out["review_id"].astype(str)
    out["dataset_key"] = out["dataset_key"].fillna("all").astype(str).str.strip().str.lower()
    out.loc[out["dataset_key"].eq(""), "dataset_key"] = "all"
    return out.drop_duplicates(subset=["review_id"], keep="first")


def _sorted_model_keys(values: list[str]) -> list[str]:
    return sorted(values, key=lambda x: MODEL_LABELS.get(x, x))


def _dataset_label(dataset_key: str) -> str:
    key = str(dataset_key).strip().lower()
    if key in {"", "all", "unknown"}:
        return "All datasets"
    return key.replace("_", " ").replace("-", " ").title()


def _topic_sort_key(topic_value: str) -> tuple[int, str]:
    value = str(topic_value).strip().lower()
    match = re.fullmatch(r"topic[_\-]?(\d+)", value)
    if match:
        return int(match.group(1)), value
    return 10**9, value


def _build_topic_display_map(
    topic_values: list[str],
    selected_model: str | None,
    selected_dataset_key: str | None,
) -> dict[str, str]:
    topics = [str(t).strip().lower() for t in topic_values]
    mapping = {topic: topic for topic in topics}
    if not topics:
        return mapping

    pack = _load_precomputed_wordcloud_terms()
    if pack.empty or not selected_model:
        for topic in topics:
            if re.fullmatch(r"topic[_\-]?(\d+)", topic):
                mapping[topic] = f"Topic {topic.split('_')[-1]}"
        return mapping

    work = pack[pack["model_key"].eq(str(selected_model).strip().lower()) & pack["ngram_n"].eq(2)].copy()
    if selected_dataset_key:
        ds_key = str(selected_dataset_key).strip().lower()
        ds_work = work[work["dataset_key"].eq(ds_key)].copy()
        if not ds_work.empty:
            work = ds_work

    if not work.empty:
        top_phrase = (
            work.groupby(["topic_proxy", "phrase"], dropna=False)["count"]
            .sum()
            .reset_index(name="count")
            .sort_values(["topic_proxy", "count"], ascending=[True, False])
            .groupby("topic_proxy", as_index=False)
            .head(1)
        )
        topic_to_phrase = dict(zip(top_phrase["topic_proxy"].astype(str), top_phrase["phrase"].astype(str)))
    else:
        topic_to_phrase = {}

    for topic in topics:
        phrase = topic_to_phrase.get(topic, "")
        match = re.fullmatch(r"topic[_\-]?(\d+)", topic)
        if match:
            base = f"Topic {int(match.group(1)):02d}"
        else:
            base = topic
        if phrase:
            mapping[topic] = f"{base} - {phrase}"
        else:
            mapping[topic] = base
    return mapping


def _topic_region_color(topic_value: str) -> str:
    topic = str(topic_value).strip().lower()
    if topic in TOPIC_REGION_COLOR_MAP:
        return TOPIC_REGION_COLOR_MAP[topic]

    hashed = pd.util.hash_pandas_object(pd.Series([topic]), index=False).iloc[0]
    idx = int(hashed % len(TOPIC_REGION_FALLBACK_COLORS))
    return TOPIC_REGION_FALLBACK_COLORS[idx]


def _cluster_region_color(cluster_value: str) -> str:
    cluster = str(cluster_value).strip().lower()
    match = re.fullmatch(r"cluster[_\-]?(\d+)", cluster)
    if match:
        idx = int(match.group(1)) % len(CLUSTER_REGION_COLORS)
        return CLUSTER_REGION_COLORS[idx]

    hashed = pd.util.hash_pandas_object(pd.Series([cluster]), index=False).iloc[0]
    idx = int(hashed % len(CLUSTER_REGION_COLORS))
    return CLUSTER_REGION_COLORS[idx]


def _split_region_components(
    coords: np.ndarray,
    eps: float,
    min_samples: int,
) -> np.ndarray:
    if coords.size == 0 or len(coords) == 0:
        return np.array([], dtype=int)

    try:
        from sklearn.cluster import DBSCAN

        labels = DBSCAN(
            eps=float(max(1e-6, eps)),
            min_samples=int(max(2, min_samples)),
            metric="euclidean",
        ).fit_predict(coords)
    except Exception:
        labels = np.zeros(len(coords), dtype=int)
    return labels


def _build_region_overlays(
    df: pd.DataFrame,
    group_by: str = "topic",
    min_group_points: int = 20,
    radius_quantile: float = 0.80,
    max_groups: int = 8,
    max_components_per_group: int = 3,
) -> list[dict]:
    if group_by == "cluster":
        group_col = "cluster_label"
        label_col = "cluster_label"
    else:
        group_col = "topic_proxy"
        label_col = "topic_display" if "topic_display" in df.columns else "topic_proxy"

    needed = {"x", "y", group_col}
    if df.empty or not needed.issubset(df.columns):
        return []

    work = df.copy()
    work["x"] = pd.to_numeric(work["x"], errors="coerce")
    work["y"] = pd.to_numeric(work["y"], errors="coerce")
    work[group_col] = work[group_col].fillna("unknown").astype(str).str.strip().str.lower()
    if group_by == "topic":
        work.loc[work[group_col].eq("unknown"), group_col] = "other"
    work[label_col] = work[label_col].fillna(work[group_col]).astype(str)

    work = work.dropna(subset=["x", "y"])
    if work.empty:
        return []

    span = float(max(work["x"].max() - work["x"].min(), work["y"].max() - work["y"].min()))
    if not np.isfinite(span) or span <= 0:
        span = 1.0

    group_counts = work[group_col].value_counts()
    if group_by == "topic":
        preferred_order = ["delivery", "product", "support", "refund", "other"]
        top_groups = [topic for topic in preferred_order if topic in group_counts.index]
        top_groups.extend([topic for topic in group_counts.index.tolist() if topic not in top_groups])
        top_groups = top_groups[: max(1, int(max_groups))]
    else:
        top_groups = (
            sorted(group_counts.index.tolist(), key=_cluster_sort_key)[: max(1, int(max_groups))]
        )

    overlays: list[dict] = []
    for group_value in top_groups:
        group_df = work[work[group_col].eq(group_value)]
        if len(group_df) < int(min_group_points):
            continue

        if group_by == "topic":
            base_label = str(group_value).strip().lower()
            color = _topic_region_color(group_value)
        else:
            base_label = str(group_value).strip()
            color = _cluster_region_color(group_value)

        component_frames: list[pd.DataFrame] = []
        coords = group_df[["x", "y"]].to_numpy(dtype=float)
        comp_min_points = int(max(8, min_group_points // 2))
        eps = float(max(1e-6, span * 0.055))
        comp_labels = _split_region_components(coords, eps=eps, min_samples=comp_min_points)
        valid_labels = pd.Series(comp_labels[comp_labels >= 0]).value_counts().index.tolist()

        if len(valid_labels) > 1:
            for comp_label in valid_labels[: max(1, int(max_components_per_group))]:
                comp_idx = np.where(comp_labels == comp_label)[0]
                comp_df = group_df.iloc[comp_idx].copy()
                if len(comp_df) >= comp_min_points:
                    component_frames.append(comp_df)

        if not component_frames:
            component_frames = [group_df]

        multi_component = len(component_frames) > 1
        for idx, comp_df in enumerate(component_frames, start=1):
            x = comp_df["x"].to_numpy(dtype=float)
            y = comp_df["y"].to_numpy(dtype=float)
            # Robust center to keep circle anchored on visible dense dots.
            cx = float(np.median(x))
            cy = float(np.median(y))
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            radius = float(np.quantile(dist, radius_quantile))
            radius = min(radius, 0.24 * span)
            if not np.isfinite(radius) or radius <= 0:
                continue

            label = f"{base_label} ({idx})" if multi_component else base_label
            overlays.append(
                {
                    "group_key": group_value,
                    "component_key": f"{group_value}__{idx}",
                    "label": label,
                    "cx": cx,
                    "cy": cy,
                    "radius": radius,
                    "color": color,
                }
            )

    return overlays


def _circle_trace_xy(cx: float, cy: float, radius: float, n_points: int = 120) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, num=max(20, int(n_points)), endpoint=True)
    xs = cx + radius * np.cos(theta)
    ys = cy + radius * np.sin(theta)
    return xs, ys


def _extract_plotly_selected_review_ids(selection_event, map_plot_df: pd.DataFrame) -> list[str]:
    """Extract selected review IDs from Plotly selection payload.

    Primary path:
    - read review_id from `selection.points[*].customdata[0]`

    Fallback path:
    - map `point_indices` to current plotted dataframe rows.
    """
    if selection_event is None:
        return []

    sel: dict = {}
    try:
        if isinstance(selection_event, dict):
            sel = selection_event.get("selection", {})
        else:
            sel_obj = getattr(selection_event, "selection", {})
            if hasattr(sel_obj, "to_dict"):
                sel = sel_obj.to_dict()
            elif isinstance(sel_obj, dict):
                sel = sel_obj
    except Exception:
        sel = {}

    if not isinstance(sel, dict):
        return []

    selected_ids: list[str] = []

    # Preferred: explicit review_id passed via custom_data in plotly trace.
    points = sel.get("points", [])
    if isinstance(points, list):
        for p in points:
            if not isinstance(p, dict):
                continue
            custom = p.get("customdata")
            rid = None
            if isinstance(custom, (list, tuple, np.ndarray)) and len(custom) > 0:
                rid = custom[0]
            elif isinstance(custom, str):
                rid = custom
            if rid is not None and str(rid).strip():
                selected_ids.append(str(rid).strip())

    if selected_ids:
        # Preserve first-seen order while deduplicating.
        return list(dict.fromkeys(selected_ids))

    # Fallback: use point_indices if available.
    idx = sel.get("point_indices", [])
    if isinstance(idx, list) and not map_plot_df.empty and "review_id" in map_plot_df.columns:
        valid_idx = [int(i) for i in idx if isinstance(i, (int, np.integer)) and 0 <= int(i) < len(map_plot_df)]
        if valid_idx:
            ids = map_plot_df.iloc[valid_idx]["review_id"].astype(str).tolist()
            return list(dict.fromkeys(ids))

    return []


def _build_focus_trend(focus_df: pd.DataFrame, freq: str = "Month") -> pd.DataFrame:
    if focus_df.empty or "review_date" not in focus_df.columns:
        return pd.DataFrame()

    work = focus_df.dropna(subset=["review_date"]).copy()
    if work.empty:
        return pd.DataFrame()

    freq_key = "W-MON" if freq == "Week" else "M"
    work["period"] = work["review_date"].dt.to_period(freq_key).dt.start_time
    out = (
        work.groupby("period")
        .agg(
            reviews=("review_id", "size"),
            negative_ratio=("sentiment", lambda s: float((s.astype(str) == "negative").mean())),
        )
        .reset_index()
        .sort_values("period")
    )
    return out


def _cosine_knn_on_projection(df: pd.DataFrame, anchor_review_id: str, k: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    work = df.reset_index(drop=True).copy()
    work["review_id"] = work["review_id"].astype(str)

    anchor_candidates = work.index[work["review_id"].eq(str(anchor_review_id))].tolist()
    if not anchor_candidates:
        return pd.DataFrame(), pd.DataFrame(), {}

    anchor_idx = int(anchor_candidates[0])
    coords = work[["x", "y"]].to_numpy(dtype=float)
    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    coords_norm = coords / norms

    anchor_vec = coords_norm[anchor_idx]
    cosine_sim = coords_norm @ anchor_vec
    cosine_dist = 1.0 - cosine_sim

    ordered_idx = np.argsort(cosine_dist)
    ordered_idx = ordered_idx[ordered_idx != anchor_idx]
    k = max(1, min(int(k), len(ordered_idx)))
    knn_idx = ordered_idx[:k]

    anchor_df = work.iloc[[anchor_idx]].copy()
    neigh_df = work.iloc[knn_idx].copy()
    neigh_df["cosine_distance"] = cosine_dist[knn_idx]
    neigh_df["cosine_similarity"] = cosine_sim[knn_idx]
    neigh_df = neigh_df.sort_values("cosine_distance", ascending=True).reset_index(drop=True)

    anchor_sentiment = str(anchor_df["sentiment"].iloc[0])
    share_same = float((neigh_df["sentiment"].astype(str) == anchor_sentiment).mean()) if len(neigh_df) else float("nan")

    sent_counts = neigh_df["sentiment"].astype(str).value_counts()
    stats = {
        "anchor_review_id": str(anchor_df["review_id"].iloc[0]),
        "anchor_sentiment": anchor_sentiment,
        "k": int(len(neigh_df)),
        "share_same_sentiment": share_same,
        "n_positive": int(sent_counts.get("positive", 0)),
        "n_neutral": int(sent_counts.get("neutral", 0)),
        "n_negative": int(sent_counts.get("negative", 0)),
    }
    return anchor_df, neigh_df, stats


def _topic_breakdown(df: pd.DataFrame, sentiment_value: str) -> pd.DataFrame:
    subset = df[df["sentiment"].astype(str).eq(sentiment_value)].copy()
    if subset.empty:
        return pd.DataFrame(columns=["topic_proxy", "volume", "share"])

    grouped = (
        subset.groupby("topic_proxy", dropna=False)
        .size()
        .reset_index(name="volume")
        .sort_values("volume", ascending=False)
        .reset_index(drop=True)
    )
    total = float(grouped["volume"].sum())
    grouped["share"] = grouped["volume"] / total if total > 0 else 0.0
    return grouped


def _negative_share_trend(df: pd.DataFrame, freq_label: str) -> pd.DataFrame:
    if "review_date" not in df.columns:
        return pd.DataFrame()

    work = df.dropna(subset=["review_date"]).copy()
    if work.empty:
        return pd.DataFrame()

    freq_key = "W-MON" if freq_label == "Week" else "M"
    work["period"] = work["review_date"].dt.to_period(freq_key).dt.start_time
    trend = (
        work.groupby("period")
        .agg(
            reviews=("review_id", "size"),
            n_negative=("sentiment", lambda s: int((s.astype(str) == "negative").sum())),
        )
        .reset_index()
        .sort_values("period")
    )
    trend["negative_share"] = trend["n_negative"] / trend["reviews"].replace(0, np.nan)
    trend["negative_share"] = trend["negative_share"].fillna(0.0)
    return trend


def _topic_volume_trend(df: pd.DataFrame, freq_label: str, top_n: int = 6) -> pd.DataFrame:
    if "review_date" not in df.columns:
        return pd.DataFrame()

    work = df.dropna(subset=["review_date"]).copy()
    if work.empty:
        return pd.DataFrame()

    freq_key = "W-MON" if freq_label == "Week" else "M"
    work["period"] = work["review_date"].dt.to_period(freq_key).dt.start_time
    top_topics = work["topic_proxy"].astype(str).value_counts().head(top_n).index.tolist()
    work = work[work["topic_proxy"].astype(str).isin(top_topics)].copy()
    if work.empty:
        return pd.DataFrame()

    topic_trend = (
        work.groupby(["period", "topic_proxy"], dropna=False)
        .size()
        .reset_index(name="reviews")
        .sort_values(["period", "reviews"], ascending=[True, False])
    )
    return topic_trend


def _extract_top_phrases(text_series: pd.Series, ngram_n: int = 2, top_n: int = 15) -> pd.DataFrame:
    token_pattern = re.compile(r"[a-zA-ZÃ€-Ã¿']+")
    phrase_counter: Counter[str] = Counter()

    for text in text_series.fillna("").astype(str):
        tokens = [t.lower() for t in token_pattern.findall(text)]
        tokens = [t for t in tokens if len(t) >= 3 and t not in DEFAULT_STOPWORDS]
        if len(tokens) < ngram_n:
            continue
        for i in range(len(tokens) - ngram_n + 1):
            phrase = " ".join(tokens[i : i + ngram_n])
            phrase_counter[phrase] += 1

    if not phrase_counter:
        return pd.DataFrame(columns=["phrase", "count"])

    top_pairs = phrase_counter.most_common(top_n)
    return pd.DataFrame(top_pairs, columns=["phrase", "count"])


def _phrase_counts_from_precomputed(
    selected_model: str | None,
    topic_value: str,
    sentiment_value: str,
    ngram_n: int,
    top_n: int = 15,
) -> pd.DataFrame:
    if not selected_model:
        return pd.DataFrame(columns=["phrase", "count"])

    pack = _load_precomputed_wordcloud_terms()
    if pack.empty:
        return pd.DataFrame(columns=["phrase", "count"])

    work = pack[
        pack["model_key"].eq(str(selected_model).strip().lower()) & pack["ngram_n"].eq(int(ngram_n))
    ].copy()
    if topic_value != "all":
        work = work[work["topic_proxy"].eq(str(topic_value).strip().lower())].copy()
    if sentiment_value != "all":
        work = work[work["sentiment"].eq(str(sentiment_value).strip().lower())].copy()

    if work.empty:
        return pd.DataFrame(columns=["phrase", "count"])

    grouped = (
        work.groupby("phrase", dropna=False)["count"]
        .sum()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(top_n)
    )
    return grouped


def _render_overview_tab(
    time_filtered_df: pd.DataFrame,
    selected_projection: str | None,
    selected_model: str | None,
    use_precomputed_terms: bool,
    phase4_df: pd.DataFrame,
    phase3_df: pd.DataFrame,
    projection_quality_df: pd.DataFrame,
    run_summary: dict,
    selected_run: str | None,
) -> None:
    st.markdown("### Business Overview")
    if time_filtered_df.empty:
        st.warning("No reviews available in the selected date window.")
        return

    total_reviews = int(len(time_filtered_df))
    n_negative = int((time_filtered_df["sentiment"].astype(str) == "negative").sum())
    n_positive = int((time_filtered_df["sentiment"].astype(str) == "positive").sum())
    neg_pct = (n_negative / total_reviews) if total_reviews else 0.0
    pos_pct = (n_positive / total_reviews) if total_reviews else 0.0

    neg_topics = _topic_breakdown(time_filtered_df, "negative")
    pos_topics = _topic_breakdown(time_filtered_df, "positive")
    top_neg = neg_topics.iloc[0]["topic_proxy"] if not neg_topics.empty else "n/a"
    top_pos = pos_topics.iloc[0]["topic_proxy"] if not pos_topics.empty else "n/a"

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(
        "Review volume",
        f"{total_reviews:,}",
        help="Total number of reviews in the active date window after sidebar filtering.",
    )
    k2.metric(
        "% negative",
        f"{neg_pct:.1%}",
        help="Share of active-window reviews labeled as negative sentiment.",
    )
    k3.metric(
        "% positive",
        f"{pos_pct:.1%}",
        help="Share of active-window reviews labeled as positive sentiment.",
    )
    k4.metric(
        "Top negative topic",
        str(top_neg),
        help="Most frequent review topic among negative reviews in the active period.",
    )
    k5.metric(
        "Top positive topic",
        str(top_pos),
        help="Most frequent review topic among positive reviews in the active period.",
    )

    left, right = st.columns(2)
    with left:
        st.markdown("#### Top Issues (Negative)")
        if neg_topics.empty:
            st.info("No negative reviews in the selected time window.")
        else:
            show_neg = neg_topics.copy().sort_values("volume", ascending=False).reset_index(drop=True)
            show_neg["share"] = (show_neg["share"] * 100).round(1)
            st.dataframe(show_neg.rename(columns={"share": "share_pct"}), width="stretch")
            top_neg_df = show_neg.head(8).copy()
            fig_neg = px.bar(
                top_neg_df,
                x="volume",
                y="topic_proxy",
                orientation="h",
                title="Negative topics by review count",
            )
            fig_neg.update_yaxes(categoryorder="total ascending")
            fig_neg.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10), yaxis_title="")
            st.plotly_chart(fig_neg, width="stretch")
    with right:
        st.markdown("#### Top Strengths (Positive)")
        if pos_topics.empty:
            st.info("No positive reviews in the selected time window.")
        else:
            show_pos = pos_topics.copy().sort_values("volume", ascending=False).reset_index(drop=True)
            show_pos["share"] = (show_pos["share"] * 100).round(1)
            st.dataframe(show_pos.rename(columns={"share": "share_pct"}), width="stretch")
            top_pos_df = show_pos.head(8).copy()
            fig_pos = px.bar(
                top_pos_df,
                x="volume",
                y="topic_proxy",
                orientation="h",
                title="Positive topics by review count",
            )
            fig_pos.update_yaxes(categoryorder="total ascending")
            fig_pos.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10), yaxis_title="")
            st.plotly_chart(fig_pos, width="stretch")

    st.markdown("#### Trends Over Time")
    trend_freq = st.radio("Trend view", options=["Week", "Month"], index=0, horizontal=True, key="trend_freq")
    trend_df = _negative_share_trend(time_filtered_df, trend_freq)
    topic_trend_df = _topic_volume_trend(time_filtered_df, trend_freq, top_n=6)

    t1, t2 = st.columns(2)
    with t1:
        if trend_df.empty:
            st.info("Negative share trend unavailable (missing dates).")
        else:
            fig_trend = px.line(
                trend_df,
                x="period",
                y="negative_share",
                markers=True,
                title=f"Negative share by {trend_freq.lower()}",
            )
            fig_trend.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10), yaxis_tickformat=".0%")
            st.plotly_chart(fig_trend, width="stretch")
    with t2:
        if topic_trend_df.empty:
            st.info("Topic volume trend unavailable (missing dates).")
        else:
            fig_topic_trend = px.area(
                topic_trend_df,
                x="period",
                y="reviews",
                color="topic_proxy",
                title=f"Topic volume by {trend_freq.lower()}",
            )
            fig_topic_trend.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig_topic_trend, width="stretch")

    st.markdown("#### Term Insights")
    topic_opts = sorted(time_filtered_df["topic_proxy"].astype(str).unique().tolist())
    sentiment_opts = sorted(time_filtered_df["sentiment"].astype(str).unique().tolist())
    c1, c2, c3 = st.columns(3)
    with c1:
        phrase_topic = st.selectbox("Topic focus", options=["all"] + topic_opts, index=0, key="phrase_topic")
    with c2:
        phrase_sent = st.selectbox("Mood focus", options=["all"] + sentiment_opts, index=0, key="phrase_sent")
    with c3:
        phrase_n = st.selectbox("Phrase length", options=[1, 2, 3], index=1, key="phrase_len")

    phrase_df = time_filtered_df.copy()
    if phrase_topic != "all":
        phrase_df = phrase_df[phrase_df["topic_proxy"].astype(str).eq(phrase_topic)].copy()
    if phrase_sent != "all":
        phrase_df = phrase_df[phrase_df["sentiment"].astype(str).eq(phrase_sent)].copy()

    phrase_counts = pd.DataFrame(columns=["phrase", "count"])
    phrase_source = "live"
    if use_precomputed_terms:
        phrase_counts = _phrase_counts_from_precomputed(
            selected_model=selected_model,
            topic_value=phrase_topic,
            sentiment_value=phrase_sent,
            ngram_n=int(phrase_n),
            top_n=15,
        )
        if not phrase_counts.empty:
            phrase_source = "precomputed"

    if phrase_counts.empty:
        phrase_counts = _extract_top_phrases(phrase_df["snippet_en"], ngram_n=int(phrase_n), top_n=15)
        phrase_source = "live"

    if phrase_counts.empty:
        st.info("No phrase data available for the selected topic/sentiment filters.")
    else:
        st.caption(f"Phrase source: `{phrase_source}`")
        phrase_plot_df = phrase_counts.sort_values("count", ascending=True)
        fig_phrase = px.bar(
            phrase_plot_df,
            x="count",
            y="phrase",
            orientation="h",
            title="Top phrases",
        )
        fig_phrase.update_layout(height=420, margin=dict(l=10, r=10, t=45, b=10), yaxis_title="")
        st.plotly_chart(fig_phrase, width="stretch")

    with st.expander("Wordcloud", expanded=False):
        if phrase_counts.empty:
            st.info("Wordcloud unavailable: no phrase frequencies in the selected subset.")
        else:
            try:
                from wordcloud import WordCloud
                import matplotlib.pyplot as plt

                freq = {
                    str(row["phrase"]): float(row["count"])
                    for _, row in phrase_counts.iterrows()
                    if str(row["phrase"]).strip() and float(row["count"]) > 0
                }
                if not freq:
                    st.info("Wordcloud unavailable: text is empty.")
                else:
                    wc = WordCloud(width=1200, height=500, background_color="white").generate_from_frequencies(freq)
                    fig_wc, ax_wc = plt.subplots(figsize=(12, 4))
                    ax_wc.imshow(wc, interpolation="bilinear")
                    ax_wc.axis("off")
                    st.pyplot(fig_wc)
            except Exception:
                st.info("Install `wordcloud` and `matplotlib` to enable this optional visual.")

    if not phase4_df.empty:
        st.markdown("#### Advanced Model Comparison")
        keep = [
            "model_key",
            "cluster_negative_ratio_std",
            "negative_homophily_k20",
            "negative_island_mass",
            "rq_signal",
        ]
        keep = [c for c in keep if c in phase4_df.columns]
        if keep:
            phase4_show = phase4_df.copy()
            if "model_key" in phase4_show.columns:
                phase4_show["model_label"] = (
                    phase4_show["model_key"].astype(str).str.lower().map(MODEL_LABELS).fillna(phase4_show["model_key"])
                )
                keep = ["model_label"] + [c for c in keep if c != "model_key"]
            display_names = {
                "model_label": "Language model",
                "cluster_negative_ratio_std": "Variation in negative rate across groups",
                "negative_homophily_k20": "Negative-neighbour concentration (k=20)",
                "negative_island_mass": "Size of isolated negative zones",
                "rq_signal": "Overall issue-signal score",
            }

            numeric_metrics = [
                c
                for c in [
                    "cluster_negative_ratio_std",
                    "negative_homophily_k20",
                    "negative_island_mass",
                ]
                if c in phase4_show.columns
            ]
            if numeric_metrics and "model_label" in phase4_show.columns:
                st.caption("Figure-based comparison across language models")
                fig_cols = st.columns(len(numeric_metrics))
                for i, metric_col in enumerate(numeric_metrics):
                    fig_metric = px.bar(
                        phase4_show,
                        x="model_label",
                        y=metric_col,
                        color="model_label",
                        title=display_names.get(metric_col, metric_col),
                        text_auto=".3f",
                    )
                    fig_metric.update_layout(
                        showlegend=False,
                        height=320,
                        margin=dict(l=10, r=10, t=50, b=10),
                        xaxis_title="",
                        yaxis_title="",
                    )
                    fig_cols[i].plotly_chart(fig_metric, width="stretch")

                if len(numeric_metrics) >= 2:
                    norm_df = phase4_show[["model_label"] + numeric_metrics].copy()
                    for metric_col in numeric_metrics:
                        series = pd.to_numeric(norm_df[metric_col], errors="coerce")
                        lo = series.min(skipna=True)
                        hi = series.max(skipna=True)
                        if pd.notna(lo) and pd.notna(hi) and hi > lo:
                            norm_df[metric_col] = (series - lo) / (hi - lo)
                        else:
                            norm_df[metric_col] = 0.5

                    norm_long = norm_df.melt(
                        id_vars="model_label",
                        value_vars=numeric_metrics,
                        var_name="metric",
                        value_name="relative_score",
                    )
                    norm_long["metric"] = norm_long["metric"].map(display_names).fillna(norm_long["metric"])
                    fig_profile = px.line_polar(
                        norm_long,
                        r="relative_score",
                        theta="metric",
                        color="model_label",
                        line_close=True,
                        markers=True,
                        title="Relative model profile (normalized)",
                    )
                    fig_profile.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
                    st.plotly_chart(fig_profile, width="stretch")

            show = phase4_show[keep].rename(columns=display_names)
            st.dataframe(show, width="stretch")

    if not phase3_df.empty:
        st.markdown("#### Phase 3 Diagnostics (analysis-v0)")
        phase3_show = phase3_df.copy()
        if "model_key" in phase3_show.columns:
            phase3_show["model_label"] = (
                phase3_show["model_key"].astype(str).str.lower().map(MODEL_LABELS).fillna(phase3_show["model_key"])
            )

        keep_cols = [
            "model_label",
            "best_k",
            "kmeans_silhouette",
            "kmeans_purity_topic_proxy",
            "kmeans_purity_sentiment",
        ]
        keep_cols = [c for c in keep_cols if c in phase3_show.columns]
        if keep_cols:
            st.dataframe(phase3_show[keep_cols], width="stretch")

        if {"model_label", "kmeans_purity_topic_proxy", "kmeans_purity_sentiment"}.issubset(phase3_show.columns):
            purity_long = phase3_show[["model_label", "kmeans_purity_topic_proxy", "kmeans_purity_sentiment"]].melt(
                id_vars="model_label",
                var_name="metric",
                value_name="value",
            )
            purity_long["metric"] = purity_long["metric"].map(
                {
                    "kmeans_purity_topic_proxy": "Topic proxy purity",
                    "kmeans_purity_sentiment": "Sentiment purity",
                }
            ).fillna(purity_long["metric"])
            fig_purity = px.bar(
                purity_long,
                x="model_label",
                y="value",
                color="metric",
                barmode="group",
                text_auto=".3f",
                title="K-means purity comparison by model",
            )
            fig_purity.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10), xaxis_title="")
            st.plotly_chart(fig_purity, width="stretch")

    if not projection_quality_df.empty:
        st.markdown("#### Projection Quality (analysis-v0)")
        quality_show = projection_quality_df.copy()
        quality_show["model_label"] = (
            quality_show["model_key"].astype(str).str.lower().map(MODEL_LABELS).fillna(quality_show["model_key"])
        )
        keep_cols = [
            "model_label",
            "projection",
            "silhouette_sentiment",
            "silhouette_topic_proxy",
            "delta_topic_minus_sentiment",
            "n_rows",
        ]
        keep_cols = [c for c in keep_cols if c in quality_show.columns]
        if keep_cols:
            st.dataframe(quality_show[keep_cols], width="stretch")

        delta_source = quality_show
        if selected_model and "model_key" in quality_show.columns:
            selected_quality = quality_show[quality_show["model_key"].eq(selected_model)].copy()
            if not selected_quality.empty:
                delta_source = selected_quality

        if {"projection", "delta_topic_minus_sentiment"}.issubset(delta_source.columns):
            fig_delta = px.bar(
                delta_source.sort_values("projection"),
                x="projection",
                y="delta_topic_minus_sentiment",
                color="model_label" if "model_label" in delta_source.columns else None,
                text_auto=".3f",
                title="Topic minus sentiment silhouette (lower means sentiment dominates)",
            )
            fig_delta.update_layout(height=340, margin=dict(l=10, r=10, t=45, b=10), xaxis_title="Projection")
            st.plotly_chart(fig_delta, width="stretch")

def _render_olist_method_output_tab(
    map_df: pd.DataFrame,
    time_filtered_df: pd.DataFrame,
    selected_projection: str | None,
    selected_model: str | None,
    phase3_df: pd.DataFrame,
    projection_quality_df: pd.DataFrame,
) -> None:
    st.markdown("### Methods and Quality Checks")
    st.caption("Advanced section for technical validation of clustering and map quality.")

    st.markdown("#### Method Flow")
    st.markdown("1. Load projected review points.")
    st.markdown("2. Attach topic labels and review snippets.")
    st.markdown("3. Evaluate cluster consistency across models.")
    st.markdown("4. Compare map quality across projection methods.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows loaded", f"{len(map_df):,}")
    c2.metric("Rows in period", f"{len(time_filtered_df):,}")
    c3.metric("Clusters", f"{time_filtered_df['cluster_label'].nunique() if not time_filtered_df.empty else 0:,}")
    c4.metric("Topics", f"{time_filtered_df['topic_proxy'].nunique() if not time_filtered_df.empty else 0:,}")

    model_key = str(selected_model or "").strip().lower()
    model_label = MODEL_LABELS.get(model_key, model_key if model_key else "n/a")

    if not phase3_df.empty and model_key:
        model_phase3 = phase3_df[phase3_df["model_key"].astype(str).str.lower().eq(model_key)].copy()
        if not model_phase3.empty:
            st.markdown("#### Clustering Diagnostics")
            row = model_phase3.iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Model", model_label)
            m2.metric(
                "Best k",
                str(int(row["best_k"])) if pd.notna(row.get("best_k")) else "n/a",
                help="Selected number of clusters from unsupervised diagnostics.",
            )
            m3.metric(
                "KMeans purity (sentiment)",
                f"{float(row['kmeans_purity_sentiment']):.3f}" if pd.notna(row.get("kmeans_purity_sentiment")) else "n/a",
                help="Cluster homogeneity with respect to sentiment labels.",
            )
            m4.metric(
                "KMeans purity (topic proxy)",
                f"{float(row['kmeans_purity_topic_proxy']):.3f}" if pd.notna(row.get("kmeans_purity_topic_proxy")) else "n/a",
                help="Cluster homogeneity with respect to topic labels.",
            )

            keep_cols = [
                "model_key",
                "best_k",
                "kmeans_silhouette",
                "kmeans_purity_sentiment",
                "kmeans_purity_topic_proxy",
            ]
            keep_cols = [c for c in keep_cols if c in model_phase3.columns]
            st.dataframe(model_phase3[keep_cols], width="stretch")

    if not projection_quality_df.empty and model_key:
        model_quality = projection_quality_df[
            projection_quality_df["model_key"].astype(str).str.lower().eq(model_key)
        ].copy()
        if not model_quality.empty:
            st.markdown("#### Projection Diagnostics")
            show_cols = [
                "model_key",
                "projection",
                "silhouette_sentiment",
                "silhouette_topic_proxy",
                "delta_topic_minus_sentiment",
                "n_rows",
            ]
            show_cols = [c for c in show_cols if c in model_quality.columns]
            st.dataframe(model_quality[show_cols].sort_values("projection"), width="stretch")

            if {"projection", "silhouette_sentiment", "silhouette_topic_proxy"}.issubset(model_quality.columns):
                long_quality = model_quality.melt(
                    id_vars="projection",
                    value_vars=["silhouette_sentiment", "silhouette_topic_proxy"],
                    var_name="metric",
                    value_name="value",
                )
                long_quality["metric"] = long_quality["metric"].map(
                    {
                        "silhouette_sentiment": "Sentiment silhouette",
                        "silhouette_topic_proxy": "Topic proxy silhouette",
                    }
                ).fillna(long_quality["metric"])
                fig_quality = px.line(
                    long_quality.sort_values("projection"),
                    x="projection",
                    y="value",
                    color="metric",
                    markers=True,
                    title=f"Projection quality by method ({model_label})",
                )
                fig_quality.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=10))
                st.plotly_chart(fig_quality, width="stretch")

    if not time_filtered_df.empty:
        st.markdown("#### Output Distributions")
        sentiment_counts = (
            time_filtered_df["sentiment"]
            .astype(str)
            .value_counts()
            .rename_axis("sentiment")
            .reset_index(name="n_reviews")
        )
        cluster_counts = (
            time_filtered_df["cluster_label"]
            .astype(str)
            .value_counts()
            .rename_axis("cluster_label")
            .reset_index(name="n_reviews")
        )
        cluster_counts["sort_key"] = cluster_counts["cluster_label"].map(_cluster_sort_key)
        cluster_counts = cluster_counts.sort_values("sort_key").drop(columns=["sort_key"])

        d1, d2 = st.columns(2)
        with d1:
            fig_sent = px.bar(
                sentiment_counts,
                x="sentiment",
                y="n_reviews",
                color="sentiment",
                color_discrete_map=SENTIMENT_COLOR_MAP,
                title="Sentiment distribution (filtered period)",
                text_auto=True,
            )
            fig_sent.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=10), xaxis_title="")
            st.plotly_chart(fig_sent, width="stretch")
        with d2:
            fig_cluster = px.bar(
                cluster_counts,
                x="cluster_label",
                y="n_reviews",
                title="Cluster size distribution (filtered period)",
                text_auto=True,
            )
            fig_cluster.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=10), xaxis_title="Cluster")
            st.plotly_chart(fig_cluster, width="stretch")


def _render_map_tab(
    map_df: pd.DataFrame,
    time_filtered_df: pd.DataFrame,
    filtered: pd.DataFrame,
    plot_df: pd.DataFrame,
    sampled: bool,
    selected_projection: str | None,
    selected_model: str | None,
    color_mode: str,
    view_mode: str,
    circle_overlay_enabled: bool,
    circle_overlay_by: str,
    depth_mode: str,
    date_start,
    date_end,
    topic_df: pd.DataFrame,
) -> None:
    st.markdown("### Detailed Review Map")
    with st.expander("How to read this map", expanded=False):
        st.markdown("1. Each dot is one review.")
        st.markdown("2. Nearby dots usually mean similar review content.")
        st.markdown("3. Dot color follows your `Color by` selection.")
        st.markdown("4. Region circles summarize areas and can overlap.")

    if date_start is not None and date_end is not None:
        st.caption(f"Active review period: {date_start} to {date_end}")

    metric_a, metric_b, metric_c, metric_d, metric_e = st.columns(5)
    metric_a.metric(
        "Rows before period filter",
        f"{len(map_df):,}",
        help="All reviews loaded for the selected map layout and language model.",
    )
    metric_b.metric(
        "Rows in selected period",
        f"{len(time_filtered_df):,}",
        help="Reviews remaining after the selected review-time window.",
    )
    metric_c.metric(
        "Rows after all filters",
        f"{len(filtered):,}",
        help="Reviews remaining after topic, mood, group, and text filters.",
    )
    metric_d.metric(
        "Points shown on map",
        f"{len(plot_df):,}",
        help="Points currently drawn on the map (may be sampled for speed).",
    )
    metric_e.metric(
        "Topics visible",
        f"{filtered['topic_proxy'].nunique() if len(filtered) else 0:,}",
        help="Number of distinct topics in the current filtered set.",
    )

    if not filtered.empty and not plot_df.empty:
        use_circle_overlay = view_mode == "2D" and bool(circle_overlay_enabled)
        if color_mode in {"Topic proxy", "Topic"}:
            color_col = "topic_display" if "topic_display" in plot_df.columns else "topic_proxy"
            color_map = None
        elif color_mode in {"Sentiment", "Mood"}:
            color_col = "sentiment"
            color_map = SENTIMENT_COLOR_MAP
        else:
            color_col = "cluster_label"
            color_map = None

        fig_title = "Review map"
        if selected_projection and selected_model:
            fig_title = f"Review map - {selected_projection} - {MODEL_LABELS.get(selected_model, selected_model)}"

        hover_data = {
            "review_id": True,
            "review_creation_date": True,
            "model_label": True,
            "topic_display": True,
            "topic_proxy": True,
            "sentiment": True,
            "cluster_label": True,
            "snippet_pt": True,
            "snippet_en": True,
            "x": ":.3f",
            "y": ":.3f",
        }
        hover_labels = {
            "review_id": "Review ID",
            "review_creation_date": "Review Date",
            "model_label": "Language Model",
            "topic_display": "Topic label",
            "topic_proxy": "Topic",
            "sentiment": "Mood",
            "cluster_label": "Review Group",
            "snippet_pt": "Snippet (PT)",
            "snippet_en": "Snippet (EN)",
            "x": "Map X coordinate",
            "y": "Map Y coordinate",
            "z_depth": "3D layer depth",
        }

        if view_mode == "3D":
            map_plot_df = plot_df.copy()
            map_plot_df["z_depth"] = _derive_depth(map_plot_df, depth_mode)
            hover_data["z_depth"] = ":.3f"
            fig = px.scatter_3d(
                map_plot_df,
                x="x",
                y="y",
                z="z_depth",
                color=color_col,
                custom_data=["review_id"],
                hover_data=hover_data,
                labels=hover_labels,
                color_discrete_map=color_map,
                opacity=0.7,
                title=f"{fig_title} (3D)",
            )
            st.caption(
                "3D depth is approximated from the selected labels (group/topic/mood), "
                "because original data contains 2D coordinates."
            )
        else:
            map_plot_df = plot_df.copy()
            fig = px.scatter(
                map_plot_df,
                x="x",
                y="y",
                color=color_col,
                custom_data=["review_id"],
                hover_data=hover_data,
                labels=hover_labels,
                color_discrete_map=color_map,
                render_mode="webgl",
                opacity=0.7,
                title=f"{fig_title} (2D)",
            )
            if use_circle_overlay:
                # Build overlays from points currently visible on map so circles match dots.
                overlay_base = plot_df.copy()
                if len(overlay_base) > 40000:
                    overlay_base = overlay_base.sample(n=40000, random_state=42).copy()
                region_group_by = "cluster" if circle_overlay_by in {"Review group", "Cluster"} else "topic"
                topic_regions = _build_region_overlays(
                    overlay_base,
                    group_by=region_group_by,
                )

                shown_legend_groups: set[str] = set()
                for region in topic_regions:
                    cx = float(region["cx"])
                    cy = float(region["cy"])
                    radius = float(region["radius"])
                    color = str(region["color"])
                    label = str(region["label"])
                    group_key = str(region.get("group_key", label))
                    legend_group = f"region_{group_key}"
                    circle_x, circle_y = _circle_trace_xy(cx, cy, radius)
                    show_legend = legend_group not in shown_legend_groups
                    if show_legend:
                        shown_legend_groups.add(legend_group)

                    fig.add_trace(
                        go.Scatter(
                            x=circle_x,
                            y=circle_y,
                            mode="lines",
                            fill="toself",
                            fillcolor=color,
                            line=dict(color=color, width=2),
                            opacity=0.12,
                            name=f"{label} region",
                            legendgroup=legend_group,
                            showlegend=show_legend,
                            hovertemplate=f"Region: {label}<br>Radius: {radius:.3f}<extra></extra>",
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=[cx],
                            y=[cy + radius],
                            mode="text",
                            text=[label],
                            textposition="top center",
                            textfont=dict(color=color, size=10),
                            legendgroup=legend_group,
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

                if topic_regions:
                    if region_group_by == "cluster":
                        st.caption(
                            "Overlay mode: points follow your `Color by` selection, and translucent circles show review-group regions."
                        )
                    else:
                        st.caption(
                            "Overlay mode: points follow your `Color by` selection, and translucent circles show topic regions "
                            "(delivery/product/support/refund/other, plus any extra active topic)."
                        )
                fig.update_yaxes(scaleanchor="x", scaleratio=1)

        fig.update_traces(marker=dict(size=6, line=dict(width=0)))
        fig.update_layout(
            height=760,
            margin=dict(l=10, r=10, t=45, b=10),
            legend_title=color_mode,
            legend=dict(groupclick="togglegroup"),
        )
        selection_event = st.plotly_chart(
            fig,
            width="stretch",
            key="projection_map_chart",
            on_select="rerun",
            selection_mode=("points", "box", "lasso"),
        )
        if sampled:
            st.caption("Rendering uses a deterministic random sample for responsiveness.")

        selected_review_ids = _extract_plotly_selected_review_ids(selection_event, map_plot_df)
        selected_points_df = pd.DataFrame()
        if selected_review_ids:
            selected_points_df = map_plot_df[
                map_plot_df["review_id"].astype(str).isin(selected_review_ids)
            ].copy()
            if not selected_points_df.empty:
                order_map = {rid: idx for idx, rid in enumerate(selected_review_ids)}
                selected_points_df["_sel_order"] = selected_points_df["review_id"].astype(str).map(order_map).fillna(10**9)
                selected_points_df = (
                    selected_points_df.sort_values("_sel_order")
                    .drop_duplicates(subset=["review_id"], keep="first")
                    .drop(columns=["_sel_order"])
                )
                selected_review_ids = selected_points_df["review_id"].astype(str).tolist()
                st.session_state["map_selected_review_ids"] = selected_review_ids
                st.session_state["map_selected_anchor_review_id"] = selected_review_ids[0]

        st.markdown("### Review Deepdive")
        st.caption("Select points on the map, or pick a topic/review group below.")

        focus_mode = st.radio(
            "Focus source",
            options=["Selected points on map", "Topic", "Review group"],
            index=0,
            key="why_focus_mode",
        )

        focus_df = pd.DataFrame()
        focus_label = "none"
        selected_review_ids_state = st.session_state.get("map_selected_review_ids", [])

        if focus_mode == "Selected points on map":
            if not selected_points_df.empty:
                focus_df = selected_points_df.copy()
                focus_label = f"{len(focus_df)} selected point(s)"
            elif selected_review_ids_state:
                focus_df = filtered[filtered["review_id"].astype(str).isin(selected_review_ids_state)].copy()
                if not focus_df.empty:
                    focus_label = f"{len(focus_df)} selected point(s)"
            if focus_df.empty:
                st.info("Use click, box, or lasso on the map to select points.")
        elif focus_mode == "Topic":
            topic_keys = sorted(filtered["topic_proxy"].astype(str).unique().tolist(), key=_topic_sort_key)
            topic_display_map = {}
            if "topic_display" in filtered.columns:
                topic_display_map = (
                    filtered[["topic_proxy", "topic_display"]]
                    .drop_duplicates(subset=["topic_proxy"], keep="first")
                    .set_index("topic_proxy")["topic_display"]
                    .astype(str)
                    .to_dict()
                )
            topic_labels = [topic_display_map.get(k, k) for k in topic_keys]
            if topic_labels:
                selected_topic_label = st.selectbox(
                    "Topic focus",
                    options=topic_labels,
                    key="why_focus_topic",
                )
                reverse_topic = {topic_display_map.get(k, k): k for k in topic_keys}
                selected_topic = reverse_topic.get(selected_topic_label, topic_keys[0])
                focus_df = filtered[filtered["topic_proxy"].astype(str).eq(str(selected_topic))].copy()
                focus_label = f"Topic: {selected_topic_label}"
        else:
            cluster_opts = sorted(filtered["cluster_label"].astype(str).unique().tolist(), key=_cluster_sort_key)
            if cluster_opts:
                selected_cluster = st.selectbox(
                    "Review group focus",
                    options=cluster_opts,
                    key="why_focus_cluster",
                )
                focus_df = filtered[filtered["cluster_label"].astype(str).eq(str(selected_cluster))].copy()
                focus_label = f"Review group: {selected_cluster}"

        if not focus_df.empty:
            volume = int(len(focus_df))
            neg_ratio = float((focus_df["sentiment"].astype(str) == "negative").mean()) if volume else 0.0
            top_topic = (
                focus_df["topic_proxy"].astype(str).mode().iloc[0]
                if "topic_proxy" in focus_df.columns and not focus_df["topic_proxy"].empty
                else "n/a"
            )
            m1, m2 = st.columns(2)
            m1.metric("Volume", f"{volume:,}")
            m2.metric("Negative ratio", f"{neg_ratio:.1%}")
            st.caption(f"Focus: {focus_label} | Top topic: {top_topic}")

            trend_df = _build_focus_trend(focus_df, freq="Month")
            if not trend_df.empty:
                fig_focus = go.Figure()
                fig_focus.add_trace(
                    go.Bar(
                        x=trend_df["period"],
                        y=trend_df["reviews"],
                        name="Reviews",
                        opacity=0.45,
                    )
                )
                fig_focus.add_trace(
                    go.Scatter(
                        x=trend_df["period"],
                        y=trend_df["negative_ratio"],
                        name="Negative ratio",
                        mode="lines+markers",
                        yaxis="y2",
                        line=dict(color="#e4572e", width=2),
                    )
                )
                fig_focus.update_layout(
                    height=280,
                    margin=dict(l=10, r=10, t=35, b=10),
                    yaxis=dict(title="Reviews"),
                    yaxis2=dict(
                        title="Negative ratio",
                        overlaying="y",
                        side="right",
                        tickformat=".0%",
                        range=[0, 1],
                    ),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                )
                st.plotly_chart(fig_focus, width="stretch")
            else:
                st.info("Trend unavailable for this focus (missing dates).")

            st.markdown("**Top review examples (EN + PT)**")
            show_df = focus_df.copy()
            if "review_date" in show_df.columns:
                show_df = show_df.sort_values("review_date", ascending=False)
            show_cols = ["review_id", "sentiment", "topic_proxy", "snippet_en", "snippet_pt"]
            show_cols = [c for c in show_cols if c in show_df.columns]
            st.dataframe(show_df[show_cols].head(5).reset_index(drop=True), width="stretch")
        else:
            st.info("No rows available for the current focus.")
    else:
        st.warning("No points matched the current map filters.")

    st.markdown("### Similar Reviews Explorer")
    st.caption(
        "Find reviews that are most similar to a selected reference review based on map coordinates."
    )
    knn_scope = st.radio(
        "Similarity search scope",
        options=[
            "All rows in selected layout/model (after period filter)",
            "Only rows currently visible after filters",
        ],
        index=0,
        horizontal=True,
        key="knn_scope_map_tab",
    )
    knn_base_df = time_filtered_df.copy() if knn_scope.startswith("All rows") else filtered.copy()

    if len(knn_base_df) > 1:
        max_k = int(min(100, len(knn_base_df) - 1))
        knn_k = st.slider(
            "Number of similar reviews (k)",
            min_value=1,
            max_value=max_k,
            value=min(20, max_k),
            step=1,
            key="knn_k_slider",
        )
        pool_n = min(500, len(knn_base_df))
        anchor_pool = (
            knn_base_df[["review_id", "sentiment", "topic_proxy", "snippet_en"]]
            .drop_duplicates(subset=["review_id"], keep="first")
            .head(pool_n)
            .copy()
        )
        anchor_pool["anchor_label"] = anchor_pool.apply(
            lambda r: f"{r['review_id']} | {r['sentiment']} | {r['topic_proxy']} | {str(r['snippet_en'])[:72]}",
            axis=1,
        )
        map_anchor_review_id = str(st.session_state.get("map_selected_anchor_review_id", "")).strip()
        if map_anchor_review_id:
            st.caption(f"Map-selected reference review: `{map_anchor_review_id}`")

        anchor_options = anchor_pool["anchor_label"].tolist()
        default_anchor_idx = 0
        if map_anchor_review_id:
            anchor_pool_ids = anchor_pool["review_id"].astype(str).tolist()
            if map_anchor_review_id in anchor_pool_ids:
                default_anchor_idx = anchor_pool_ids.index(map_anchor_review_id)

        selected_anchor_label = st.selectbox(
            "Reference review (quick pick)",
            options=anchor_options,
            index=default_anchor_idx if anchor_options else 0,
            key="knn_anchor_quickpick",
        )
        quick_anchor_id = str(selected_anchor_label).split(" | ", 1)[0] if selected_anchor_label else ""
        manual_anchor_id = st.text_input(
            "Or paste review ID",
            value="",
            placeholder="e.g. 61a271eca0ed85d04936382e3b9829a9",
            key="knn_anchor_manual",
        ).strip()
        use_map_selected_anchor = False
        if map_anchor_review_id:
            use_map_selected_anchor = st.checkbox(
                "Use selected map point as reference review",
                value=True,
                key="knn_use_map_selected_anchor",
            )

        if manual_anchor_id:
            anchor_review_id = manual_anchor_id
        elif use_map_selected_anchor and map_anchor_review_id:
            anchor_review_id = map_anchor_review_id
        else:
            anchor_review_id = quick_anchor_id

        anchor_df, neigh_df, knn_stats = _cosine_knn_on_projection(knn_base_df, anchor_review_id, knn_k)
        if anchor_df.empty or neigh_df.empty:
            st.warning("Reference review not found in the selected scope, or no similar reviews are available.")
        else:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric(
                "Reference mood",
                str(knn_stats.get("anchor_sentiment", "n/a")),
                help="Mood label for the selected reference review.",
            )
            m2.metric(
                "k",
                str(knn_stats.get("k", 0)),
                help="Number of similar reviews returned for the selected reference review.",
            )
            share_same = knn_stats.get("share_same_sentiment", float("nan"))
            m3.metric(
                "Same-mood share",
                f"{share_same:.2f}" if pd.notna(share_same) else "n/a",
                help="Fraction of similar reviews with the same mood as the reference review.",
            )
            m4.metric(
                "Positive / Neutral",
                f"{knn_stats.get('n_positive', 0)} / {knn_stats.get('n_neutral', 0)}",
                help="Count of positive and neutral reviews in the similar-review set.",
            )
            m5.metric(
                "Negative",
                str(knn_stats.get("n_negative", 0)),
                help="Count of negative reviews in the similar-review set.",
            )

            local_df = pd.concat(
                [
                    anchor_df.assign(knn_role="Anchor"),
                    neigh_df.assign(knn_role="Neighbour"),
                ],
                ignore_index=True,
            )
            local_df["knn_size"] = np.where(local_df["knn_role"].eq("Anchor"), 22, 12)
            knn_proj_label = selected_projection if selected_projection is not None else "Map"
            fig_knn = px.scatter(
                local_df,
                x="x",
                y="y",
                color="sentiment",
                symbol="knn_role",
                size="knn_size",
                hover_data={
                    "review_id": True,
                    "sentiment": True,
                    "topic_proxy": True,
                    "cosine_distance": ":.4f",
                    "snippet_en": True,
                    "snippet_pt": True,
                    "x": ":.3f",
                    "y": ":.3f",
                },
                labels={
                    "review_id": "Review ID",
                    "sentiment": "Mood",
                    "topic_proxy": "Topic",
                    "cosine_distance": "Cosine Distance",
                    "snippet_en": "Snippet (EN)",
                    "snippet_pt": "Snippet (PT)",
                    "x": "Map X coordinate",
                    "y": "Map Y coordinate",
                },
                color_discrete_map=SENTIMENT_COLOR_MAP,
                title=f"Similar reviews around selected review (k={knn_stats['k']}, {knn_proj_label})",
                opacity=0.8,
            )
            fig_knn.update_layout(height=520, margin=dict(l=10, r=10, t=45, b=10))
            st.plotly_chart(fig_knn, width="stretch")

            neigh_cols = [
                "review_id",
                "cosine_distance",
                "cosine_similarity",
                "sentiment",
                "topic_proxy",
                "snippet_en",
                "snippet_pt",
            ]
            neigh_cols = [c for c in neigh_cols if c in neigh_df.columns]
            st.dataframe(neigh_df[neigh_cols].reset_index(drop=True), width="stretch")
    else:
        st.caption("Similar Reviews Explorer unavailable: not enough rows in the selected scope.")

    if not plot_df.empty:
        download_df = plot_df.copy()
        if selected_projection is not None:
            download_df["projection"] = selected_projection
        if view_mode == "3D":
            download_df["z_depth"] = _derive_depth(download_df.copy(), depth_mode).values
            download_df["depth_mode"] = depth_mode

        csv_cols = [
            "x",
            "y",
            "z_depth",
            "depth_mode",
            "model_key",
            "cluster_label",
            "topic_proxy",
            "sentiment",
            "review_id",
            "review_creation_date",
            "snippet_pt",
            "snippet_en",
        ]
        if "projection" in download_df.columns:
            csv_cols = ["projection"] + csv_cols
        csv_cols = [c for c in csv_cols if c in download_df.columns]
        csv_bytes = download_df[csv_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download rendered points (CSV)",
            data=csv_bytes,
            file_name="semantic_map_filtered.csv",
            mime="text/csv",
            key="download_rendered_points",
        )
        if st.button("Prepare map snapshot (PNG)", key="prepare_map_png_btn"):
            try:
                st.session_state["map_png_bytes"] = fig.to_image(format="png", width=1500, height=900, scale=2)
                st.success("PNG snapshot prepared.")
            except Exception:
                st.session_state.pop("map_png_bytes", None)
                st.caption("PNG export unavailable. Install `kaleido` to enable map image download.")

        png_bytes = st.session_state.get("map_png_bytes")
        if isinstance(png_bytes, (bytes, bytearray)) and len(png_bytes) > 0:
            st.download_button(
                "Download map snapshot (PNG)",
                data=png_bytes,
                file_name="semantic_map_snapshot.png",
                mime="image/png",
                key="download_map_png",
            )

        with st.expander("Rendered rows (PT + EN snippets)", expanded=False):
            show_cols = [
                "review_id",
                "review_creation_date",
                "model_label",
                "cluster_label",
                "topic_proxy",
                "sentiment",
                "snippet_pt",
                "snippet_en",
            ]
            show_cols = [c for c in show_cols if c in plot_df.columns]
            st.dataframe(plot_df[show_cols].reset_index(drop=True), width="stretch")

def main() -> None:
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)
    _render_stakeholder_intro()
    filters_summary_placeholder = st.empty()
    data_badge_placeholder = st.empty()

    projection_root, projection_files = _discover_projection_files()
    phase3_df = _load_phase3_model_comparison()
    projection_quality_df = _load_projection_quality_table()
    if not projection_files:
        st.error(
            "No analysis-v0 projection outputs found. Expected:\n"
            "- `data/projection_views/viz_*.csv` (preferred) or `data/projection_views/viz_*.parquet`"
        )
        return

    projection_options = list(projection_files.keys())
    default_projection_idx = projection_options.index("UMAP") if "UMAP" in projection_options else 0

    with st.sidebar.expander("Display settings", expanded=False):
        st.markdown("- Scope is fixed to Olist.")
        st.markdown("- Use beginner mode for simple navigation.")
        st.markdown("- Open advanced settings only when you need deeper diagnostics.")
    _render_sidebar_glossary()
    show_advanced_settings = st.sidebar.checkbox(
        "Show advanced settings",
        value=False,
        help="Off by default for non-technical users. Turn on to choose map version/model and 3D options.",
        key="show_advanced_settings",
    )
    date_start = None
    date_end = None
    time_window = "All time"

    selected_projection = projection_options[default_projection_idx]
    if show_advanced_settings:
        selected_projection = st.sidebar.selectbox(
            "Map version",
            options=projection_options,
            index=default_projection_idx,
            help="Different map versions arrange the same reviews differently.",
            key="map_version",
        )
    else:
        st.sidebar.caption(f"Map version: {selected_projection} (default)")

    projection_path = projection_files[selected_projection]
    projection_df = _load_projection_points(projection_path)
    if projection_df.empty:
        st.error(f"Selected map version file is empty or invalid: `{projection_path.name}`")
        return

    model_keys = _sorted_model_keys(projection_df["model_key"].dropna().astype(str).unique().tolist())
    if not model_keys:
        st.error("No language model keys found in the selected map layout.")
        return

    model_labels = [MODEL_LABELS.get(k, k) for k in model_keys]
    label_to_key = {MODEL_LABELS.get(k, k): k for k in model_keys}
    default_model_key = "multilingual_minilm" if "multilingual_minilm" in model_keys else model_keys[0]
    default_model_labels = [MODEL_LABELS.get(default_model_key, default_model_key)]
    if show_advanced_settings:
        active_model_labels = st.sidebar.multiselect(
            "Language model",
            options=model_labels,
            default=default_model_labels,
            help="Choose one or more models. This changes map geometry and diagnostics.",
            key="active_model_labels",
        )
        if not active_model_labels:
            active_model_labels = default_model_labels
    else:
        active_model_labels = default_model_labels
    active_model_keys = [label_to_key[label] for label in active_model_labels if label in label_to_key]
    if not active_model_keys:
        active_model_keys = model_keys
    selected_model = active_model_keys[0] if len(active_model_keys) == 1 else None
    map_df = projection_df[projection_df["model_key"].isin(active_model_keys)].copy()

    if map_df.empty:
        st.error("No map points available after loading source data.")
        return

    dataset_lookup_df = _load_dataset_lookup(None)
    if not dataset_lookup_df.empty and "review_id" in map_df.columns:
        enriched = map_df.merge(dataset_lookup_df, on="review_id", how="left", suffixes=("", "_lookup"))
        if "dataset_key_lookup" in enriched.columns:
            needs_backfill = enriched["dataset_key"].isin(["all", "unknown", ""])
            enriched.loc[needs_backfill, "dataset_key"] = enriched.loc[needs_backfill, "dataset_key_lookup"].fillna(
                enriched.loc[needs_backfill, "dataset_key"]
            )
            enriched = enriched.drop(columns=["dataset_key_lookup"])
        map_df = enriched

    if "dataset_key" not in map_df.columns:
        map_df["dataset_key"] = "olist"
    map_df["dataset_key"] = map_df["dataset_key"].fillna("unknown").astype(str).str.strip().str.lower()

    if map_df["dataset_key"].eq("olist").any():
        map_df = map_df[map_df["dataset_key"].eq("olist")].copy()
    else:
        st.warning("No explicit `olist` dataset tag found in the loaded projection bundle. Using all loaded rows as Olist scope.")
        map_df["dataset_key"] = "olist"

    if map_df.empty:
        st.error("No Olist rows available after filtering.")
        return

    active_dataset_key = "olist"
    active_dataset_label = "Olist"
    if len(active_model_labels) == len(model_labels):
        model_scope_label = "All models"
    else:
        model_scope_label = ", ".join(active_model_labels)
    st.caption(
        f"Showing `{len(map_df):,}` Olist reviews | Map version: `{selected_projection}` | Model scope: `{model_scope_label}`"
    )
    if selected_projection == "tSNE" and len(map_df) <= (3000 * max(1, len(active_model_keys))):
        st.info(
            "tSNE view is sampled in the current export. Use UMAP or PCA if you need broader coverage."
        )

    date_values = (
        map_df["review_date"].dropna()
        if "review_date" in map_df.columns
        else pd.Series(dtype="datetime64[ns]")
    )
    if not date_values.empty:
        min_date = date_values.min().date()
        max_date = date_values.max().date()
        time_window = st.sidebar.selectbox(
            "Review period",
            options=["All time", "Last 30 days", "Last quarter (90 days)", "Last 365 days", "Custom range"],
            index=0,
            help="Filter reviews by creation date before map and metrics are computed.",
            key="time_window",
        )
        if time_window == "Last 30 days":
            date_start = max(min_date, (pd.Timestamp(max_date) - pd.Timedelta(days=29)).date())
            date_end = max_date
        elif time_window == "Last quarter (90 days)":
            date_start = max(min_date, (pd.Timestamp(max_date) - pd.Timedelta(days=89)).date())
            date_end = max_date
        elif time_window == "Last 365 days":
            date_start = max(min_date, (pd.Timestamp(max_date) - pd.Timedelta(days=364)).date())
            date_end = max_date
        elif time_window == "Custom range":
            default_start = max(min_date, (pd.Timestamp(max_date) - pd.Timedelta(days=89)).date())
            picked = st.sidebar.date_input(
                "Date range",
                value=(default_start, max_date),
                min_value=min_date,
                max_value=max_date,
                key="date_range",
            )
            if isinstance(picked, tuple) and len(picked) == 2:
                date_start, date_end = picked
            elif isinstance(picked, list) and len(picked) == 2:
                date_start, date_end = picked[0], picked[1]
            else:
                date_start, date_end = min_date, max_date
            if date_start > date_end:
                date_start, date_end = date_end, date_start
        st.sidebar.caption(f"Available period: {min_date} to {max_date}")
    else:
        st.sidebar.caption("Review period filter unavailable: no dates found in current data source.")

    sentiments = sorted(map_df["sentiment"].unique().tolist())
    clusters = sorted(map_df["cluster_label"].unique().tolist(), key=_cluster_sort_key)
    topics = sorted(map_df["topic_proxy"].fillna("unknown").astype(str).unique().tolist(), key=_topic_sort_key)
    topic_display_map = _build_topic_display_map(
        topic_values=topics,
        selected_model=selected_model,
        selected_dataset_key=active_dataset_key,
    )
    map_df["topic_display"] = map_df["topic_proxy"].map(topic_display_map).fillna(map_df["topic_proxy"]).astype(str)
    topic_options = [topic_display_map.get(topic_key, topic_key) for topic_key in topics]
    topic_label_to_key = {topic_display_map.get(topic_key, topic_key): topic_key for topic_key in topics}

    if show_advanced_settings:
        view_mode = st.sidebar.radio(
            "Map mode",
            ["2D", "3D"],
            index=0,
            help="2D is easiest to interpret. 3D adds a vertical layer by selected labels.",
            key="view_mode",
        )
    else:
        view_mode = "2D"
    circle_overlay_enabled = False
    circle_overlay_by = "Topic"
    if view_mode == "2D":
        circle_overlay_enabled = st.sidebar.checkbox(
            "Show region circles",
            value=False,
            help="Overlay translucent region circles while keeping the standard 2D scatter.",
            key="circle_overlay_enabled",
        )
        circle_overlay_by = st.sidebar.radio(
            "Circle by",
            options=["Topic", "Review group"],
            index=0,
            disabled=not circle_overlay_enabled,
            help="Choose whether region circles summarize topic areas or cluster areas.",
            key="circle_overlay_by",
        )
    depth_mode = "Review group"
    if view_mode == "3D":
        depth_mode = st.sidebar.selectbox(
            "3D layer style",
            options=["Review group", "Topic", "Mood"],
            index=0,
            help="Chooses what creates vertical separation in 3D view.",
            key="depth_mode",
        )

    color_mode = st.sidebar.radio(
        "Color by",
        ["Topic", "Mood", "Review group"],
        index=0,
        help="Controls dot colors only; circle overlays are configured separately.",
        key="color_mode",
    )
    active_topic_labels = st.sidebar.multiselect(
        "Topic",
        topic_options,
        default=topic_options,
        help="Filter which topic labels are included in the map.",
        key="active_topic_labels",
    )
    active_topics = [topic_label_to_key[label] for label in active_topic_labels if label in topic_label_to_key]
    if not active_topics:
        active_topics = topics
    active_sentiments = st.sidebar.multiselect(
        "Mood",
        sentiments,
        default=sentiments,
        help="Filter reviews by sentiment label.",
        key="active_sentiments",
    )
    active_clusters = st.sidebar.multiselect(
        "Review group",
        clusters,
        default=clusters,
        help="Filter reviews by cluster IDs.",
        key="active_clusters",
    )
    query = st.sidebar.text_input(
        "Search in review text (PT or EN)",
        value="",
        help="Keyword filter over Portuguese and English snippets.",
        key="query_text",
    ).strip()

    time_filtered_df = map_df.copy()
    if date_start is not None and date_end is not None and "review_date" in time_filtered_df.columns:
        start_ts = pd.Timestamp(date_start)
        end_ts = pd.Timestamp(date_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        time_filtered_df = time_filtered_df[
            time_filtered_df["review_date"].between(start_ts, end_ts, inclusive="both")
        ].copy()

    filtered = time_filtered_df[
        time_filtered_df["topic_proxy"].isin(active_topics)
        & time_filtered_df["sentiment"].isin(active_sentiments)
        & time_filtered_df["cluster_label"].isin(active_clusters)
    ].copy()

    if query:
        pt_hit = filtered["snippet_pt"].str.contains(query, case=False, na=False)
        en_hit = filtered["snippet_en"].str.contains(query, case=False, na=False)
        filtered = filtered[pt_hit | en_hit].copy()

    show_all_points = True
    if show_advanced_settings:
        show_all_points = st.sidebar.checkbox(
            "Show all filtered points",
            value=True,
            help="Enabled by default so the map reflects the full filtered dataset.",
            key="show_all_points",
        )
    max_points = len(filtered)
    if not show_all_points and len(filtered) > 500:
        max_points = st.sidebar.slider(
            "Max points on map",
            min_value=500,
            max_value=len(filtered),
            value=len(filtered),
            step=500,
            key="max_points_slider",
        )

    sampled = False
    plot_df = filtered.copy()
    if max_points > 0 and len(filtered) > max_points:
        plot_df = filtered.sample(n=max_points, random_state=42).copy()
        sampled = True

    period_label = "All time"
    if date_start is not None and date_end is not None:
        period_label = f"{date_start} to {date_end}"
    query_label = query if query else "none"
    active_filters_summary = (
        f"Active filters: period `{period_label}` | topics `{len(active_topics)}/{len(topics)}` | "
        f"mood `{len(active_sentiments)}/{len(sentiments)}` | review groups `{len(active_clusters)}/{len(clusters)}` | "
        f"search `{query_label}`"
    )
    filters_summary_placeholder.caption(active_filters_summary)

    latest_ts = map_df["review_date"].dropna().max() if "review_date" in map_df.columns and map_df["review_date"].notna().any() else pd.NaT
    if pd.notna(latest_ts):
        days_old = int((pd.Timestamp.today().normalize() - pd.Timestamp(latest_ts).normalize()).days)
        freshness_label = f"Last review date `{pd.Timestamp(latest_ts).date()}` ({days_old} days old)"
    else:
        freshness_label = "Last review date unavailable"

    overlay_note = "no overlay"
    if view_mode == "2D" and circle_overlay_enabled:
        overlay_mode_label = "review groups" if circle_overlay_by == "Review group" else "topics"
        overlay_note = f"heuristic circles by {overlay_mode_label}"

    sample_note = "full points shown" if not sampled else f"sampled to `{len(plot_df):,}` points"
    rows_used_label = (
        f"Rows used: loaded `{len(map_df):,}` -> period `{len(time_filtered_df):,}` -> "
        f"filtered `{len(filtered):,}` -> rendered `{len(plot_df):,}` ({sample_note})"
    )
    data_badge_placeholder.caption(f"Data freshness: {freshness_label} | {rows_used_label} | Overlay: {overlay_note}")

    use_precomputed_terms = not _load_precomputed_wordcloud_terms().empty
    tab_summary, tab_map, tab_method = st.tabs(["Business Summary", "Projection Map", "Methods (Advanced)"])
    with tab_summary:
        _render_overview_tab(
            time_filtered_df=time_filtered_df,
            selected_projection=selected_projection,
            selected_model=selected_model,
            use_precomputed_terms=use_precomputed_terms,
            phase4_df=pd.DataFrame(),
            phase3_df=pd.DataFrame(),
            projection_quality_df=pd.DataFrame(),
            run_summary={},
            selected_run=None,
        )

    with tab_method:
        _render_olist_method_output_tab(
            map_df=map_df,
            time_filtered_df=time_filtered_df,
            selected_projection=selected_projection,
            selected_model=selected_model,
            phase3_df=phase3_df,
            projection_quality_df=projection_quality_df,
        )

    with tab_map:
        _render_map_tab(
            map_df=map_df,
            time_filtered_df=time_filtered_df,
            filtered=filtered,
            plot_df=plot_df,
            sampled=sampled,
            selected_projection=selected_projection,
            selected_model=selected_model,
            color_mode=color_mode,
            view_mode=view_mode,
            circle_overlay_enabled=circle_overlay_enabled,
            circle_overlay_by=circle_overlay_by,
            depth_mode=depth_mode,
            date_start=date_start,
            date_end=date_end,
            topic_df=pd.DataFrame(),
        )


if __name__ == "__main__":
    main()

