from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from .paths import data_dir, outputs_dir


DEFAULT_RAW_PATH = data_dir() / "RawData.csv"


def load_raw_data(path: Optional[str | Path] = None) -> pd.DataFrame:
    path = Path(path) if path else DEFAULT_RAW_PATH
    return pd.read_csv(path)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=True)
    return path


def save_json(data: dict, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return path


def default_outputs_dir() -> Path:
    out = outputs_dir()
    return ensure_dir(out)
