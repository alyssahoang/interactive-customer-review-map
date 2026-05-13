"""Projection engines (PCA, tSNE, UMAP) used for semantic map visualization."""

from __future__ import annotations

from dataclasses import dataclass
import inspect

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


@dataclass
class ProjectionEngine:
    """Generate low-dimensional projections from embedding matrices."""

    random_state: int = 42

    def pca(self, embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
        reducer = PCA(n_components=n_components, random_state=self.random_state)
        return reducer.fit_transform(embeddings)

    def tsne(
        self,
        embeddings: np.ndarray,
        n_components: int = 2,
        perplexity: float = 30.0,
        n_iter: int = 1000,
        max_iter: int | None = None,
    ) -> np.ndarray:
        # scikit-learn renamed TSNE argument n_iter -> max_iter in newer versions.
        resolved_max_iter = int(max_iter if max_iter is not None else n_iter)

        tsne_kwargs = {
            "n_components": n_components,
            "perplexity": perplexity,
            "init": "pca",
            "random_state": self.random_state,
            "learning_rate": "auto",
        }
        param_names = inspect.signature(TSNE.__init__).parameters
        if "max_iter" in param_names:
            tsne_kwargs["max_iter"] = resolved_max_iter
        else:
            tsne_kwargs["n_iter"] = resolved_max_iter

        reducer = TSNE(**tsne_kwargs)
        return reducer.fit_transform(embeddings)

    def umap(
        self,
        embeddings: np.ndarray,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
    ) -> np.ndarray:
        try:
            import umap
        except ImportError as exc:
            raise ImportError("Install `umap-learn` to use UMAP projections.") from exc

        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="cosine",
            random_state=self.random_state,
        )
        return reducer.fit_transform(embeddings)

    def project(self, embeddings: np.ndarray, method: str, **kwargs) -> np.ndarray:
        method = method.lower()
        if method == "pca":
            return self.pca(embeddings, **kwargs)
        if method == "tsne":
            return self.tsne(embeddings, **kwargs)
        if method == "umap":
            return self.umap(embeddings, **kwargs)
        raise ValueError("Unknown projection method. Use one of: pca, tsne, umap.")
