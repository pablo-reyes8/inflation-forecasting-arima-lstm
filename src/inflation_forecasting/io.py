"""Legacy flat import path.

Prefer `inflation_forecasting.datasets.io` for active code.
"""

from .datasets.io import (
    DEFAULT_CLEANED_PATH,
    DEFAULT_RAW_PATH,
    SUPPORTED_DATA_EXTENSIONS,
    default_outputs_dir,
    ensure_dir,
    load_raw_data,
    read_tabular_data,
    save_dataframe,
    save_json,
    save_yaml,
)

__all__ = [
    "DEFAULT_CLEANED_PATH",
    "DEFAULT_RAW_PATH",
    "SUPPORTED_DATA_EXTENSIONS",
    "default_outputs_dir",
    "ensure_dir",
    "load_raw_data",
    "read_tabular_data",
    "save_dataframe",
    "save_json",
    "save_yaml",
]
