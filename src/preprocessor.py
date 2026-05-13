"""Text cleaning and lightweight sentiment/topic proxy preprocessing for Olist reviews."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable

import numpy as np
import pandas as pd


def _load_pt_stopwords() -> set[str]:
    try:
        import nltk
        from nltk.corpus import stopwords

        try:
            words = set(stopwords.words("portuguese"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            words = set(stopwords.words("portuguese"))
        return words
    except Exception:
        # Minimal fallback list to keep the project runnable without NLTK assets.
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
            "esse",
            "esta",
            "eu",
            "foi",
            "mas",
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
            "sem",
            "ser",
            "tem",
            "um",
            "uma",
        }


class _PortugueseHeuristicStemmer:
    """Lightweight fallback stemmer for environments without NLTK.

    It applies conservative suffix stripping to reduce inflection variance.
    """

    _suffixes: tuple[str, ...] = (
        "mente",
        "ções",
        "cao",
        "ção",
        "sões",
        "sao",
        "são",
        "idades",
        "idade",
        "ismos",
        "ismo",
        "istas",
        "ista",
        "ivos",
        "ivas",
        "ivo",
        "iva",
        "amentos",
        "imento",
        "imentos",
        "amento",
        "adoras",
        "adores",
        "adora",
        "ador",
        "antes",
        "ante",
        "mente",
        "logias",
        "logia",
        "ções",
        "ção",
        "ados",
        "adas",
        "ado",
        "ada",
        "idos",
        "idas",
        "ido",
        "ida",
        "ando",
        "endo",
        "indo",
        "ar",
        "er",
        "ir",
        "es",
        "s",
    )

    def stem(self, token: str) -> str:
        t = token.strip()
        if len(t) <= 4:
            return t
        for suf in self._suffixes:
            if t.endswith(suf) and len(t) - len(suf) >= 3:
                return t[: -len(suf)]
        return t


def _load_portuguese_stemmer():
    """Return a Portuguese stemmer object with a `.stem()` method.

    Preferred: NLTK Snowball stemmer.
    Fallback: local heuristic stemmer to keep pipeline deterministic/offline.
    """
    try:
        from nltk.stem.snowball import SnowballStemmer

        return SnowballStemmer("portuguese")
    except Exception:  # pragma: no cover - dependency/import guard
        return _PortugueseHeuristicStemmer()


@dataclass
class PortugueseTextPreprocessor:
    lowercase: bool = True
    remove_numbers: bool = True
    remove_stopwords: bool = True
    use_stemming: bool = False
    min_token_length: int = 2
    extra_stopwords: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.stopwords = _load_pt_stopwords()
        self.stopwords.update(self.extra_stopwords)
        self.stemmer = _load_portuguese_stemmer() if self.use_stemming else None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", text)

    def preprocess_text(self, text: str) -> str:
        if not isinstance(text, str):
            text = "" if pd.isna(text) else str(text)

        if self.lowercase:
            text = text.lower()

        if self.remove_numbers:
            text = re.sub(r"\d+", " ", text)

        text = re.sub(r"\s+", " ", text).strip()
        tokens = self._tokenize(text)

        cleaned_tokens = []
        for token in tokens:
            if len(token) < self.min_token_length:
                continue
            if self.remove_stopwords and token in self.stopwords:
                continue
            if self.stemmer is not None:
                token = self.stemmer.stem(token)
            cleaned_tokens.append(token)

        return " ".join(cleaned_tokens)

    def transform_series(self, text_series: pd.Series) -> pd.Series:
        return text_series.fillna("").astype(str).map(self.preprocess_text)


@dataclass
class PortugueseSentimentLabeler:
    positive_words: set[str] = field(
        default_factory=lambda: {
            "bom",
            "boa",
            "excelente",
            "otimo",
            "ótimo",
            "perfeito",
            "rapido",
            "rápido",
            "recomendo",
            "gostei",
            "satisfeito",
            "qualidade",
            "amei",
            "super",
        }
    )
    negative_words: set[str] = field(
        default_factory=lambda: {
            "ruim",
            "pessimo",
            "péssimo",
            "atraso",
            "atrasado",
            "demora",
            "decepcionado",
            "horrivel",
            "horrível",
            "problema",
            "quebrado",
            "cancelado",
            "nunca",
            "insatisfeito",
        }
    )

    @staticmethod
    def _score_to_label(score: float | int | None) -> str:
        if score is None or pd.isna(score):
            return "neutral"
        if score <= 2:
            return "negative"
        if score >= 4:
            return "positive"
        return "neutral"

    def _lexicon_score(self, text: str) -> int:
        tokens = text.split()
        pos_hits = sum(token in self.positive_words for token in tokens)
        neg_hits = sum(token in self.negative_words for token in tokens)
        return pos_hits - neg_hits

    def label_text(self, text: str, review_score: float | int | None = None) -> str:
        base_label = self._score_to_label(review_score)
        lex_score = self._lexicon_score(text or "")

        if lex_score >= 2:
            return "positive"
        if lex_score <= -2:
            return "negative"
        return base_label

    def annotate(
        self,
        df: pd.DataFrame,
        text_col: str = "clean_text",
        score_col: str = "review_score",
        output_col: str = "sentiment_label",
    ) -> pd.DataFrame:
        labeled = df.copy()
        labeled[output_col] = [
            self.label_text(text=row_text, review_score=row_score)
            for row_text, row_score in zip(labeled[text_col], labeled[score_col])
        ]
        return labeled

    def distribution(self, sentiment_series: Iterable[str]) -> pd.Series:
        series = pd.Series(list(sentiment_series))
        return series.value_counts(normalize=True).sort_index()
