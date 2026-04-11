from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from ..paths import data_dir, outputs_dir
from .preprocessing import RAW_DATA_REQUIRED_COLUMNS, validate_required_columns


DEFAULT_RAW_PATH = data_dir() / "RawData.csv"
DEFAULT_CLEANED_PATH = data_dir() / "Data Cleaned.xlsx"
SUPPORTED_DATA_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def _read_excel(source: Any) -> pd.DataFrame:
    try:
        return pd.read_excel(source)
    except ImportError as exc:
        raise ImportError(
            "Reading Excel datasets requires an engine such as `openpyxl` for .xlsx files "
            "or `xlrd` for legacy .xls files."
        ) from exc


def _read_tabular(source: Any, suffix: str) -> pd.DataFrame:
    normalized = suffix.lower()
    if normalized == ".csv":
        return pd.read_csv(source)
    if normalized in {".xlsx", ".xls"}:
        return _read_excel(source)
    raise ValueError(f"Unsupported dataset format '{normalized}'. Expected one of: {sorted(SUPPORTED_DATA_EXTENSIONS)}")


def read_tabular_data(file_name: str, payload: bytes) -> pd.DataFrame:
    suffix = Path(file_name).suffix.lower()
    return _read_tabular(BytesIO(payload), suffix)


def load_raw_data(
    path: str | Path | None = None,
    *,
    validate: bool = True,
    required_columns: tuple[str, ...] = RAW_DATA_REQUIRED_COLUMNS,
) -> pd.DataFrame:
    dataset_path = Path(path) if path else DEFAULT_RAW_PATH
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = _read_tabular(dataset_path, dataset_path.suffix)
    if validate:
        validate_required_columns(df, required_columns)
    return df


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_dataframe(df: pd.DataFrame, path: str | Path, *, index: bool = True) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=index)
    return path


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_to_serializable(data), file, indent=2, sort_keys=True)
    return path


def save_yaml(data: dict[str, Any], path: str | Path) -> Path:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to save YAML manifests. Install with `pip install PyYAML`.") from exc

    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(_to_serializable(data), file, allow_unicode=False, sort_keys=False)
    return path


def default_outputs_dir() -> Path:
    out = outputs_dir()
    return ensure_dir(out)
