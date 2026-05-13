"""Data quality diagnostics used before embedding and clustering stages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DataQualityAuditor:
    """Dataset-quality checks inspired by EDA-first notebook workflows."""

    def dataset_overview(self, df: pd.DataFrame, id_col: str | None = None) -> pd.DataFrame:
        n_rows, n_cols = df.shape
        duplicate_rows = int(df.duplicated().sum())
        duplicate_ratio = duplicate_rows / n_rows if n_rows else np.nan

        distinct_ids = np.nan
        if id_col is not None and id_col in df.columns:
            distinct_ids = int(df[id_col].nunique(dropna=True))

        report = pd.DataFrame(
            [
                {"metric": "n_rows", "value": n_rows},
                {"metric": "n_columns", "value": n_cols},
                {"metric": "duplicate_rows", "value": duplicate_rows},
                {"metric": "duplicate_ratio", "value": round(float(duplicate_ratio), 6)},
                {"metric": f"distinct_{id_col}" if id_col else "distinct_id", "value": distinct_ids},
            ]
        )
        return report

    def missingness_report(self, df: pd.DataFrame, sort_desc: bool = True) -> pd.DataFrame:
        miss = df.isna().mean().rename("missing_ratio").reset_index()
        miss = miss.rename(columns={"index": "column"})
        miss["missing_ratio"] = miss["missing_ratio"].astype(float)
        if sort_desc:
            miss = miss.sort_values("missing_ratio", ascending=False)
        return miss.reset_index(drop=True)

    def text_quality_report(
        self,
        df: pd.DataFrame,
        text_col: str = "review_text",
        score_col: str = "review_score",
        id_col: str = "review_id",
    ) -> pd.DataFrame:
        text = df[text_col].fillna("").astype(str)
        non_empty = text.str.strip().ne("")
        lengths = text.str.len()
        token_lengths = text.str.split().map(len)

        dup_text_ratio = (
            float(text[non_empty].duplicated().mean())
            if int(non_empty.sum()) > 0
            else np.nan
        )

        rows = [
            {"metric": "rows_total", "value": int(len(df))},
            {"metric": "rows_non_empty_text", "value": int(non_empty.sum())},
            {"metric": "text_non_empty_ratio", "value": round(float(non_empty.mean()), 6)},
            {"metric": "text_duplicate_ratio_non_empty", "value": round(float(dup_text_ratio), 6)},
            {"metric": "text_length_mean", "value": round(float(lengths.mean()), 3)},
            {"metric": "text_length_median", "value": round(float(lengths.median()), 3)},
            {"metric": "token_length_mean", "value": round(float(token_lengths.mean()), 3)},
        ]

        if id_col in df.columns:
            rows.append({"metric": "distinct_ids", "value": int(df[id_col].nunique(dropna=True))})

        if score_col in df.columns:
            rows.append({"metric": "review_score_missing_ratio", "value": round(float(df[score_col].isna().mean()), 6)})

        return pd.DataFrame(rows)

    def score_distribution(self, df: pd.DataFrame, score_col: str = "review_score") -> pd.DataFrame:
        dist = (
            df[score_col]
            .value_counts(dropna=False)
            .rename_axis(score_col)
            .rename("count")
            .reset_index()
            .sort_values(score_col)
            .reset_index(drop=True)
        )
        dist["ratio"] = dist["count"] / dist["count"].sum()
        return dist
