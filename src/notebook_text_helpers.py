"""Notebook text utilities kept reusable in the src layer."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


def _candidate_data_dirs() -> list[Path]:
    """Resolve likely data directories for notebook execution contexts."""
    cwd = Path.cwd().resolve()
    return [
        cwd / "data",
        cwd.parent / "data",
        cwd / "submission" / "data",
        cwd.parent / "submission" / "data",
    ]


@lru_cache(maxsize=1)
def _translation_maps() -> tuple[dict[str, str], dict[str, str]]:
    """
    Build cached PT->EN lookup maps from shipped Olist files.
    Returns:
      - review_id -> english text
      - portuguese text -> english text (first non-empty match)
    """
    cols = ["review_id", "review_comment_message"]
    for data_dir in _candidate_data_dirs():
        src_path = data_dir / "olist_order_reviews_dataset.csv"
        en_path = data_dir / "olist_order_reviews_dataset_translated.csv"
        if not src_path.exists() or not en_path.exists():
            continue
        try:
            src = pd.read_csv(src_path, usecols=cols)
            en = pd.read_csv(en_path, usecols=cols)
        except Exception:
            continue

        src = src.rename(columns={"review_comment_message": "review_text_pt"})
        en = en.rename(columns={"review_comment_message": "review_text_en"})
        merged = src.merge(en, on="review_id", how="left")
        merged["review_text_pt"] = merged["review_text_pt"].fillna("").astype(str)
        merged["review_text_en"] = merged["review_text_en"].fillna("").astype(str)
        merged = merged[merged["review_text_pt"].str.strip() != ""].copy()

        id_to_en = (
            merged.drop_duplicates("review_id", keep="first")
            .set_index("review_id")["review_text_en"]
            .to_dict()
        )
        pt_to_en_map = (
            merged[merged["review_text_en"].str.strip() != ""]
            .drop_duplicates("review_text_pt", keep="first")
            .set_index("review_text_pt")["review_text_en"]
            .to_dict()
        )
        return id_to_en, pt_to_en_map

    return {}, {}


def pt_to_en(text: str, review_id: str | None = None) -> str:
    """Lookup-based PT->EN helper using shipped Kaggle translated file."""
    if not isinstance(text, str) or not text.strip():
        return ""
    id_to_en, pt_to_en_map = _translation_maps()
    if review_id is not None:
        key = str(review_id)
        en = id_to_en.get(key, "")
        if isinstance(en, str) and en.strip():
            return en
    return pt_to_en_map.get(text, "")


def to_en_safe(text: str, review_id: str | None = None) -> str:
    """Safe wrapper around lookup-based translation."""
    if not isinstance(text, str) or not text.strip():
        return ""
    try:
        return pt_to_en(text, review_id=review_id)
    except Exception:
        return ""


def load_portuguese_stopwords() -> set[str]:
    """Load PT stopwords from NLTK with lightweight fallback."""
    try:
        import nltk
        from nltk.corpus import stopwords

        try:
            return set(stopwords.words("portuguese"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            return set(stopwords.words("portuguese"))
    except Exception:
        return {
            "a",
            "ao",
            "aos",
            "as",
            "com",
            "da",
            "das",
            "de",
            "do",
            "dos",
            "e",
            "em",
            "na",
            "nas",
            "no",
            "nos",
            "o",
            "os",
            "para",
            "por",
            "que",
            "se",
            "um",
            "uma",
        }


def top_terms(
    text_series: pd.Series,
    ngram_range: tuple[int, int] = (1, 1),
    min_df: int = 20,
    top_n: int = 20,
    pt_stopwords: set[str] | None = None,
) -> pd.DataFrame:
    """Compute top n-grams by raw frequency with robust defaults."""
    from sklearn.feature_extraction.text import CountVectorizer

    stop_words = list(pt_stopwords or load_portuguese_stopwords())

    text_series = text_series.fillna("").astype(str)
    text_series = text_series[text_series.str.strip() != ""]
    if len(text_series) == 0:
        return pd.DataFrame(columns=["term", "count"])
    try:
        vec = CountVectorizer(
            stop_words=stop_words,
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=20000,
        )
        X = vec.fit_transform(text_series)
    except ValueError:
        vec = CountVectorizer(
            stop_words=stop_words,
            ngram_range=ngram_range,
            min_df=2,
            max_features=20000,
        )
        X = vec.fit_transform(text_series)

    freqs = np.asarray(X.sum(axis=0)).ravel()
    terms = np.array(vec.get_feature_names_out())
    order = np.argsort(freqs)[::-1][:top_n]
    return pd.DataFrame({"term": terms[order], "count": freqs[order].astype(int)})


def text_stats(series: pd.Series) -> pd.Series:
    """Vocabulary-size and average token-length summary."""
    tokens = series.str.split().explode()
    tokens = tokens[tokens.notna() & (tokens != "")]
    return pd.Series(
        {
            "vocab_size": tokens.nunique(),
            "avg_tokens_per_review": series.str.split().map(len).mean(),
        }
    )


def score_review(tokens: list[str], valence_map: dict[str, float]) -> float:
    """Lexicon-based review score from token list."""
    return float(sum(valence_map.get(tok, 0.0) for tok in tokens))


def sample_examples(
    df: pd.DataFrame,
    rating: str,
    lexicon: str,
    n: int = 5,
) -> pd.DataFrame:
    """Sample bilingual examples for rating/lexicon combinations."""
    subset = df[(df["rating_group"] == rating) & (df["lexicon_group"] == lexicon)].copy()
    base_cols = ["review_score", "rating_group", "lexicon_group", "lexicon_score", "review_text"]
    if "review_id" in subset.columns:
        base_cols = ["review_id"] + base_cols
    subset = subset[base_cols].head(n)
    if "review_id" in subset.columns:
        subset["review_en"] = subset.apply(
            lambda r: to_en_safe(r["review_text"], review_id=str(r["review_id"])),
            axis=1,
        )
    else:
        subset["review_en"] = subset["review_text"].map(to_en_safe)
    return subset
