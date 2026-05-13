"""Embedding model interfaces and concrete encoders for review text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


class TextEmbedder:
    """Simple interface for embedding text into dense vectors."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


def _batch_iter(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


@dataclass
class HuggingFaceMeanPoolEmbedder(TextEmbedder):
    """Mean-pooled transformer encoder loaded lazily from HuggingFace."""

    model_name: str
    batch_size: int = 32
    max_length: int = 256
    device: str | None = None

    def __post_init__(self) -> None:
        self.tokenizer = None
        self.model = None
        self._torch = None

    def _lazy_load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Install `transformers` and `torch` to use HuggingFaceMeanPoolEmbedder."
            ) from exc

        self._torch = torch
        runtime_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(runtime_device)
        self.model.eval()
        self.device = runtime_device

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.empty((0, 0), dtype=np.float32)

        self._lazy_load()
        vectors = []

        with self._torch.no_grad():
            for batch in _batch_iter(list(texts), self.batch_size):
                encoded = self.tokenizer(
                    list(batch),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                output = self.model(**encoded)
                token_embeddings = output.last_hidden_state
                attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size())

                sum_embeddings = (token_embeddings * attention_mask).sum(dim=1)
                token_counts = attention_mask.sum(dim=1).clamp(min=1e-9)
                mean_embeddings = sum_embeddings / token_counts

                vectors.append(mean_embeddings.cpu().numpy().astype(np.float32))

        return np.vstack(vectors)


@dataclass
class SentenceTransformerEmbedder(TextEmbedder):
    """SentenceTransformer wrapper with configurable batching."""

    model_name: str
    batch_size: int = 64
    normalize_embeddings: bool = True
    device: str | None = None

    def __post_init__(self) -> None:
        self.model = None

    def _lazy_load(self) -> None:
        if self.model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Install `sentence-transformers` to use SentenceTransformerEmbedder."
            ) from exc

        self.model = SentenceTransformer(self.model_name, device=self.device)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.empty((0, 0), dtype=np.float32)

        self._lazy_load()
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)


@dataclass
class OpenCLIPTextEmbedder(TextEmbedder):
    """OpenCLIP text encoder used for modality-transfer semantic checks."""

    model_name: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    batch_size: int = 64
    normalize_embeddings: bool = True
    device: str | None = None

    def __post_init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self._torch = None

    def _lazy_load(self) -> None:
        if self.model is not None:
            return
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise ImportError(
                "Install `open_clip_torch` and `torch` to use OpenCLIPTextEmbedder."
            ) from exc

        runtime_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, _, _ = open_clip.create_model_and_transforms(
            model_name=self.model_name,
            pretrained=self.pretrained,
            device=runtime_device,
        )
        model.eval()
        self.model = model
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.device = runtime_device
        self._torch = torch

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.empty((0, 0), dtype=np.float32)

        self._lazy_load()
        vectors = []

        with self._torch.no_grad():
            for batch in _batch_iter(list(texts), self.batch_size):
                tokens = self.tokenizer(list(batch)).to(self.device)
                batch_vectors = self.model.encode_text(tokens)
                if self.normalize_embeddings:
                    batch_vectors = batch_vectors / batch_vectors.norm(dim=-1, keepdim=True)
                vectors.append(batch_vectors.cpu().numpy().astype(np.float32))

        return np.vstack(vectors)


class EmbeddingFactory:
    """Factory that resolves model keys to embedding backends."""

    PRESETS = {
        "bertimbau": {
            "type": "hf",
            "model_name": "neuralmind/bert-base-portuguese-cased",
        },
        "multilingual_minilm": {
            "type": "st",
            "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        },
        "openclip_text": {
            "type": "openclip",
            "model_name": "ViT-B-32",
            "pretrained": "laion2b_s34b_b79k",
        },
    }

    @classmethod
    def available_model_keys(cls) -> list[str]:
        return sorted(cls.PRESETS.keys())

    @classmethod
    def create(cls, model_key: str, device: str | None = None) -> TextEmbedder:
        if model_key not in cls.PRESETS:
            raise KeyError(f"Unknown model key: {model_key}. Available: {cls.available_model_keys()}")

        preset = cls.PRESETS[model_key]
        preset_type = preset["type"]

        if preset_type == "hf":
            return HuggingFaceMeanPoolEmbedder(model_name=preset["model_name"], device=device)
        if preset_type == "st":
            return SentenceTransformerEmbedder(model_name=preset["model_name"], device=device)
        if preset_type == "openclip":
            return OpenCLIPTextEmbedder(
                model_name=preset["model_name"],
                pretrained=preset["pretrained"],
                device=device,
            )
        raise ValueError(f"Unsupported preset type: {preset_type}")
