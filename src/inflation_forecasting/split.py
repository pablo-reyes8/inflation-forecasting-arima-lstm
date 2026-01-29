from __future__ import annotations

from typing import Tuple

import pandas as pd


def train_test_split_series(series: pd.Series, test_size: float = 0.2) -> Tuple[pd.Series, pd.Series]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    n = len(series)
    split_idx = int(n * (1 - test_size))
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]
    return train, test


def train_val_test_split_series(
    series: pd.Series,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    if not abs((train_size + val_size + test_size) - 1.0) < 1e-8:
        raise ValueError("train_size + val_size + test_size must equal 1")
    n = len(series)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)
    train = series.iloc[:train_end]
    val = series.iloc[train_end:val_end]
    test = series.iloc[val_end:]
    return train, val, test
