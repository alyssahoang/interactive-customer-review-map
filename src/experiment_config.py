"""Experiment configuration objects for notebook orchestration and scripts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """Central configuration for Olist sentiment-geometry experiments."""

    # Data controls (None = full file)
    audit_sample_size: int | None = None
    sample_size: int | None = None
    min_chars: int | None = None
    min_retention_target: float = 0.90

    # Reproducibility
    random_state: int = 42
    stability_seeds: tuple[int, ...] = (11, 21, 42, 84)

    # Embeddings
    model_keys: tuple[str, ...] = ("bertimbau", "multilingual_minilm", "openclip_text")
    model_key: str = "multilingual_minilm"

    # Projection methods
    projection_methods: tuple[str, ...] = ("pca", "umap", "tsne")
    projection_method: str = "umap"

    # Clustering search
    k_grid: tuple[int, ...] = (2, 4, 6, 8, 10, 12)

    def validate(self) -> None:
        """Validate selected model and projection keys."""
        if self.model_key not in self.model_keys:
            raise ValueError(f"model_key={self.model_key!r} must be one of {self.model_keys}")
        if self.projection_method not in self.projection_methods:
            raise ValueError(
                f"projection_method={self.projection_method!r} must be one of {self.projection_methods}"
            )

