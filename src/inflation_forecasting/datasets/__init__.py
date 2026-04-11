"""Dataset access, preprocessing, features and time-based splits.

This namespace intentionally avoids the ambiguous top-level `Data/` asset folder.
Use `inflation_forecasting.datasets` for code and `Data/` for repository data assets.
"""

from .features import make_lag_features
from .io import (
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
from .preprocessing import (
    RAW_DATA_REQUIRED_COLUMNS,
    add_quarterly_date,
    filter_state,
    interpolate_series,
    prepare_state_series,
    summary_stats,
    to_time_series,
    validate_required_columns,
)
from .splits import resolve_split_index, train_test_split_series, train_val_test_split_series

__all__ = [
    "DEFAULT_CLEANED_PATH",
    "DEFAULT_RAW_PATH",
    "RAW_DATA_REQUIRED_COLUMNS",
    "SUPPORTED_DATA_EXTENSIONS",
    "add_quarterly_date",
    "default_outputs_dir",
    "ensure_dir",
    "filter_state",
    "interpolate_series",
    "load_raw_data",
    "make_lag_features",
    "prepare_state_series",
    "read_tabular_data",
    "resolve_split_index",
    "save_dataframe",
    "save_json",
    "save_yaml",
    "summary_stats",
    "to_time_series",
    "train_test_split_series",
    "train_val_test_split_series",
    "validate_required_columns",
]
