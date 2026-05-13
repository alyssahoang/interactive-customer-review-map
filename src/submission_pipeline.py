"""Reproducible entrypoint helpers for submission artifact generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

import pandas as pd


@dataclass
class SubmissionPipeline:
    """Run and validate key submission pipeline steps."""

    project_root: Path

    @property
    def scripts_dir(self) -> Path:
        return self.project_root / "scripts"

    @property
    def projection_dir(self) -> Path:
        return self.project_root / "data" / "phase3_multi" / "projection_views"

    @property
    def report_dir(self) -> Path:
        return self.project_root / "report"

    @property
    def artifacts_run_root(self) -> Path:
        return self.project_root / "artifacts" / "analysis-v1"

    def _run_dirs(self) -> list[Path]:
        if not self.artifacts_run_root.exists():
            return []
        return sorted([p for p in self.artifacts_run_root.iterdir() if p.is_dir()], key=lambda p: p.name)

    def resolve_latest_run_dir(self) -> Path | None:
        """Return latest non-empty run directory, if available."""
        candidates = [p for p in self._run_dirs() if any(p.glob("*"))]
        return candidates[-1] if candidates else None

    def validate_bundle(self) -> pd.DataFrame:
        """Validate required submission files for first-time reproducibility."""
        required_paths = [
            self.project_root / "data" / "olist_order_reviews_dataset.csv",
            self.project_root / "data" / "olist_order_reviews_dataset_translated.csv",
            self.project_root / "data" / "processed" / "reviews_preprocessed.pkl",
            self.project_root / "data" / "phase3_multi" / "phase3_model_comparison.csv",
            self.project_root / "data" / "phase3_multi" / "projection_quality_all_models.csv",
            self.projection_dir / "viz_pca.parquet",
            self.projection_dir / "viz_tsne.parquet",
            self.projection_dir / "viz_umap.parquet",
            self.project_root / "interactive-map-lite" / "app.py",
            self.project_root / "notebook" / "olist_negative_sentiment_geometry_analysis.ipynb",
            self.project_root / "report" / "olist_negative_sentiment_geometry_report.tex",
        ]

        latest_run = self.resolve_latest_run_dir()
        if latest_run is not None:
            required_paths.extend(
                [
                    latest_run / "phase3_model_comparison.csv",
                    latest_run / "phase4_model_comparison.csv",
                    latest_run / "kmeans_scores_all_models.csv",
                    latest_run / "topic_table.csv",
                    latest_run / "rq2_pairwise_delta_ci.csv",
                ]
            )

        rows = []
        for path in required_paths:
            exists = path.exists()
            rows.append(
                {
                    "kind": "required",
                    "exists": bool(exists),
                    "path": str(path),
                    "size_mb": round(path.stat().st_size / (1024 * 1024), 3) if exists and path.is_file() else 0.0,
                }
            )

        optional_paths = [
            self.project_root / "data" / "phase3_multi" / "projection_views" / "wordcloud_terms.parquet",
            self.project_root / "data" / "phase3_multi" / "projection_views" / "review_text_lookup.csv",
            self.project_root / "data" / "phase3_multi" / "projection_views" / "topic_proxy_lookup.parquet",
        ]
        for path in optional_paths:
            exists = path.exists()
            rows.append(
                {
                    "kind": "optional",
                    "exists": bool(exists),
                    "path": str(path),
                    "size_mb": round(path.stat().st_size / (1024 * 1024), 3) if exists and path.is_file() else 0.0,
                }
            )

        return pd.DataFrame(rows)

    def assert_bundle(self) -> None:
        """Raise an error when required reproducibility files are missing."""
        status = self.validate_bundle()
        missing = status[(status["kind"] == "required") & (~status["exists"])]
        if not missing.empty:
            missing_paths = "\n".join(f"- {p}" for p in missing["path"].tolist())
            raise FileNotFoundError(
                "Submission bundle is incomplete. Missing required files:\n"
                f"{missing_paths}\n\n"
                "Run `python scripts/run_submission_pipeline.py` to rebuild packable artifacts."
            )

    def run_projection_rebuild(self) -> None:
        script = self.scripts_dir / "rebuild_projection_views.py"
        # Rebuild projection-view parquet files from existing cached coords by default.
        # This keeps the submission reproducible without shipping large embedding caches.
        cmd = [sys.executable, str(script), "--skip-tsne-compute"]
        subprocess.run(cmd, check=True, cwd=str(self.project_root))

    def run_report_pack(self, run_dir: Path | None = None) -> None:
        script = self.scripts_dir / "prepare_report_outputs_v2.py"
        cmd = [sys.executable, str(script)]
        if run_dir is None:
            run_dir = self.resolve_latest_run_dir()
        if run_dir is not None:
            cmd.extend(["--run-dir", str(run_dir)])
        subprocess.run(cmd, check=True, cwd=str(self.project_root))

    def describe_outputs(self) -> pd.DataFrame:
        """Return a compact status table used by notebook demo cells."""
        expected = {
            "viz_pca": self.projection_dir / "viz_pca.parquet",
            "viz_tsne": self.projection_dir / "viz_tsne.parquet",
            "viz_umap": self.projection_dir / "viz_umap.parquet",
            "topic_proxy_lookup": self.projection_dir / "topic_proxy_lookup.parquet",
            "review_text_lookup": self.projection_dir / "review_text_lookup.csv",
            "wordcloud_terms": self.projection_dir / "wordcloud_terms.parquet",
        }
        rows = []
        for key, path in expected.items():
            rows.append(
                {
                    "artifact_key": key,
                    "exists": bool(path.exists()),
                    "path": str(path),
                    "size_mb": round(path.stat().st_size / (1024 * 1024), 3) if path.exists() else 0.0,
                }
            )
        return pd.DataFrame(rows).sort_values("artifact_key").reset_index(drop=True)
