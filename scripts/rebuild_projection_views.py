from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

MODEL_KEYS = ["bertimbau", "multilingual_minilm", "openclip_text"]
PROJECTIONS = [("pca", "PCA"), ("tsne", "tSNE"), ("umap", "UMAP")]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_phase3_root() -> Path:
    return _project_root() / "data" / "phase3_multi"


def _default_embeddings_root() -> Path:
    return _project_root() / "data" / "embeddings"


def _default_projection_root() -> Path:
    return _default_phase3_root() / "projection_views"


def _parse_models(raw: str | None) -> list[str]:
    if not raw:
        return MODEL_KEYS.copy()
    out = [m.strip().lower() for m in raw.split(",") if m.strip()]
    if not out:
        raise ValueError("No valid model keys passed to --models.")
    return out


def _find_embedding_file(embeddings_root: Path, model_key: str) -> Path:
    matches = sorted(embeddings_root.glob(f"{model_key}_*_l2.npy"))
    if not matches:
        raise FileNotFoundError(
            f"No L2 embedding file found for model '{model_key}' under: {embeddings_root}"
        )
    return matches[-1]


def _load_label_index(model_dir: Path) -> pd.DataFrame:
    label_path = model_dir / "label_index.parquet"
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label index: {label_path}")

    label_df = pd.read_parquet(label_path)
    required = {"review_id", "sentiment_label", "topic_proxy"}
    missing = required - set(label_df.columns)
    if missing:
        raise ValueError(f"{label_path} is missing required columns: {sorted(missing)}")

    if "kmeans_cluster" in label_df.columns:
        cluster_col = "kmeans_cluster"
    elif "cluster_id" in label_df.columns:
        cluster_col = "cluster_id"
    else:
        cluster_candidates = [c for c in label_df.columns if c.endswith("_cluster")]
        if not cluster_candidates:
            raise ValueError(f"{label_path} has no cluster column (expected kmeans_cluster or *_cluster).")
        cluster_col = cluster_candidates[0]

    out = pd.DataFrame(
        {
            "review_id": label_df["review_id"].astype(str),
            "sentiment": label_df["sentiment_label"].fillna("other").astype(str).str.strip().str.lower(),
            "topic_proxy": label_df["topic_proxy"].fillna("unknown").astype(str).str.strip().str.lower(),
            "cluster": label_df[cluster_col].fillna("unknown").astype(str),
        }
    )

    known_sentiments = {"negative", "neutral", "positive"}
    out.loc[~out["sentiment"].isin(known_sentiments), "sentiment"] = "other"
    out.loc[out["topic_proxy"].eq(""), "topic_proxy"] = "unknown"
    out.loc[out["cluster"].eq(""), "cluster"] = "unknown"
    return out


def _compute_tsne_coords(
    embeddings: np.ndarray,
    random_state: int,
    perplexity: float,
    max_iter: int,
    pre_pca_dims: int,
    n_jobs: int,
) -> np.ndarray:
    work = np.asarray(embeddings, dtype=np.float32)

    if pre_pca_dims > 0 and work.shape[1] > pre_pca_dims:
        pca = PCA(n_components=pre_pca_dims, random_state=random_state)
        work = pca.fit_transform(work).astype(np.float32, copy=False)
        print(f"    - pre-PCA -> {work.shape}")

    tsne_kwargs = dict(
        n_components=2,
        perplexity=float(perplexity),
        init="pca",
        learning_rate="auto",
        random_state=int(random_state),
        method="barnes_hut",
        angle=0.5,
        max_iter=int(max_iter),
        verbose=1,
    )
    if "n_jobs" in inspect.signature(TSNE).parameters:
        tsne_kwargs["n_jobs"] = int(n_jobs)
    tsne = TSNE(**tsne_kwargs)
    coords = tsne.fit_transform(work)
    return coords.astype(np.float32, copy=False)


def _ensure_tsne_cache(
    phase3_root: Path,
    embeddings_root: Path,
    model_keys: list[str],
    random_state: int,
    perplexity: float,
    max_iter: int,
    pre_pca_dims: int,
    n_jobs: int,
    force_tsne: bool,
    compute_missing_tsne: bool,
) -> None:
    for model_key in model_keys:
        model_dir = phase3_root / model_key
        cache_path = model_dir / "coords_tsne.npy"

        if cache_path.exists() and not force_tsne:
            print(f"[{model_key}] tSNE cache exists -> {cache_path.name}")
            continue

        if not compute_missing_tsne and not cache_path.exists():
            print(f"[{model_key}] tSNE cache missing and compute disabled -> skipping")
            continue

        label_df = _load_label_index(model_dir)
        emb_path = _find_embedding_file(embeddings_root, model_key)
        embeddings = np.load(emb_path, mmap_mode="r")

        if embeddings.shape[0] != len(label_df):
            raise ValueError(
                f"{model_key}: embeddings rows={embeddings.shape[0]} but label rows={len(label_df)}."
            )

        print(f"[{model_key}] computing full tSNE from {emb_path.name} | shape={embeddings.shape}")
        coords = _compute_tsne_coords(
            embeddings=embeddings,
            random_state=random_state,
            perplexity=perplexity,
            max_iter=max_iter,
            pre_pca_dims=pre_pca_dims,
            n_jobs=n_jobs,
        )
        if coords.shape[0] != len(label_df) or coords.shape[1] < 2:
            raise ValueError(f"{model_key}: invalid tSNE output shape={coords.shape}")

        np.save(cache_path, coords[:, :2].astype(np.float32, copy=False))
        print(f"[{model_key}] saved -> {cache_path}")


def _build_projection_frame(phase3_root: Path, model_key: str, method_key: str) -> pd.DataFrame:
    model_dir = phase3_root / model_key
    label_df = _load_label_index(model_dir)
    coords_path = model_dir / f"coords_{method_key}.npy"
    if not coords_path.exists():
        raise FileNotFoundError(f"Missing projection cache: {coords_path}")

    coords = np.load(coords_path)
    if coords.ndim != 2 or coords.shape[0] != len(label_df) or coords.shape[1] < 2:
        raise ValueError(
            f"{coords_path} has invalid shape={coords.shape}; expected ({len(label_df)}, 2+)."
        )

    frame = label_df.copy()
    frame["x"] = coords[:, 0].astype(np.float32, copy=False)
    frame["y"] = coords[:, 1].astype(np.float32, copy=False)
    frame["model_key"] = model_key
    frame["snippet"] = ""
    return frame[["x", "y", "cluster", "sentiment", "review_id", "model_key", "topic_proxy", "snippet"]]


def _write_projection_views(
    phase3_root: Path,
    projection_root: Path,
    model_keys: list[str],
) -> None:
    projection_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []

    for method_key, method_label in PROJECTIONS:
        frames: list[pd.DataFrame] = []
        for model_key in model_keys:
            frame = _build_projection_frame(phase3_root=phase3_root, model_key=model_key, method_key=method_key)
            frames.append(frame)

        out_df = pd.concat(frames, ignore_index=True)
        out_path = projection_root / f"viz_{method_key}.parquet"
        out_df.to_parquet(out_path, index=False)
        manifest_rows.append(
            {
                "method": method_label,
                "n_rows": int(len(out_df)),
                "path": str(out_path.resolve()),
            }
        )
        print(f"[{method_label}] saved {len(out_df):,} rows -> {out_path}")

    manifest_path = projection_root / "projection_view_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"Manifest saved -> {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild data/phase3_multi/projection_views/viz_*.parquet from cached coords files "
            "and optionally compute missing full coords_tsne.npy from embedding caches."
        )
    )
    parser.add_argument("--phase3-root", type=Path, default=_default_phase3_root())
    parser.add_argument("--embeddings-root", type=Path, default=_default_embeddings_root())
    parser.add_argument("--projection-root", type=Path, default=_default_projection_root())
    parser.add_argument("--models", type=str, default=",".join(MODEL_KEYS))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--tsne-perplexity", type=float, default=35.0)
    parser.add_argument("--tsne-max-iter", type=int, default=1200)
    parser.add_argument("--tsne-pre-pca-dims", type=int, default=50)
    parser.add_argument("--tsne-n-jobs", type=int, default=1)
    parser.add_argument(
        "--skip-tsne-compute",
        action="store_true",
        help="Do not compute missing coords_tsne.npy; only rebuild from existing caches.",
    )
    parser.add_argument(
        "--force-tsne",
        action="store_true",
        help="Recompute coords_tsne.npy even when cache already exists.",
    )
    args = parser.parse_args()

    phase3_root = args.phase3_root.resolve()
    embeddings_root = args.embeddings_root.resolve()
    projection_root = args.projection_root.resolve()
    model_keys = _parse_models(args.models)

    if not phase3_root.exists():
        raise FileNotFoundError(f"phase3 root not found: {phase3_root}")
    compute_missing_tsne = not args.skip_tsne_compute
    if not embeddings_root.exists():
        if compute_missing_tsne:
            print(
                f"Warning: embeddings root not found ({embeddings_root}); "
                "disabling tSNE recompute and using existing cached coords only."
            )
        compute_missing_tsne = False

    print(f"phase3_root      : {phase3_root}")
    print(f"embeddings_root  : {embeddings_root} (exists={embeddings_root.exists()})")
    print(f"projection_root  : {projection_root}")
    print(f"model_keys       : {model_keys}")

    _ensure_tsne_cache(
        phase3_root=phase3_root,
        embeddings_root=embeddings_root,
        model_keys=model_keys,
        random_state=args.random_state,
        perplexity=args.tsne_perplexity,
        max_iter=args.tsne_max_iter,
        pre_pca_dims=args.tsne_pre_pca_dims,
        n_jobs=args.tsne_n_jobs,
        force_tsne=args.force_tsne,
        compute_missing_tsne=compute_missing_tsne,
    )
    _write_projection_views(
        phase3_root=phase3_root,
        projection_root=projection_root,
        model_keys=model_keys,
    )


if __name__ == "__main__":
    main()
