"""Dataset loaders for Olist review files used in the P15 analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

REVIEW_COLUMNS = [
    "review_id",
    "order_id",
    "review_score",
    "review_comment_title",
    "review_comment_message",
    "review_creation_date",
    "review_answer_timestamp",
]


@dataclass(frozen=True)
class OlistPaths:
    """Filesystem contract for required Olist source files."""

    data_dir: Path
    reviews_filename: str = "olist_order_reviews_dataset.csv"

    @property
    def reviews_path(self) -> Path:
        return self.data_dir / self.reviews_filename


class OlistReviewLoader:
    """Load and prepare Olist review text in Portuguese."""

    def __init__(self, data_dir: str | Path) -> None:
        self.paths = OlistPaths(data_dir=Path(data_dir))
        if not self.paths.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.paths.data_dir}")
        if not self.paths.reviews_path.exists():
            raise FileNotFoundError(f"Reviews file not found: {self.paths.reviews_path}")

    def load_reviews(
        self,
        require_text: bool = True,
        min_chars: int = 5,
        include_title: bool = False,
        sample_size: Optional[int] = None,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Load review records and expose a unified `review_text` column."""
        df = pd.read_csv(self.paths.reviews_path, usecols=REVIEW_COLUMNS)

        for dt_col in ("review_creation_date", "review_answer_timestamp"):
            df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")

        title_text = df["review_comment_title"].fillna("").astype(str).str.strip()
        message_text = df["review_comment_message"].fillna("").astype(str).str.strip()

        if include_title:
            df["review_text"] = (title_text + ". " + message_text).str.strip(". ").str.strip()
        else:
            df["review_text"] = message_text

        if require_text:
            df = df[df["review_text"].str.len() >= min_chars].copy()

        df = df.drop_duplicates(subset=["review_id"]).reset_index(drop=True)

        if sample_size is not None and sample_size < len(df):
            frac = sample_size / len(df)
            if df["review_score"].nunique() > 1:
                df = (
                    df.groupby("review_score", group_keys=False)
                    .sample(frac=frac, random_state=random_state)
                    .reset_index(drop=True)
                )
            else:
                df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

        return df
