"""Reproducibility utilities for deterministic runs and artifact persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


@dataclass
class ArtifactStore:
    root_dir: Path
    run_name: str = "analysis-v1"
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d-%H%M%S"))

    @property
    def run_dir(self) -> Path:
        return self.root_dir / self.run_name / self.timestamp

    def ensure(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self.run_dir

    def save_json(self, payload: dict[str, Any], filename: str) -> Path:
        out = self.ensure() / filename
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return out

    def save_dataframe(self, df: pd.DataFrame, filename: str, index: bool = False) -> Path:
        out = self.ensure() / filename
        df.to_csv(out, index=index)
        return out

    def save_text(self, content: str, filename: str) -> Path:
        out = self.ensure() / filename
        out.write_text(content, encoding="utf-8")
        return out
