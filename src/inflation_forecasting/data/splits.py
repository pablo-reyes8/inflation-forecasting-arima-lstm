from __future__ import annotations

from typing import Tuple

import pandas as pd


def resolve_split_index(
    length: int,
    test_size: float | int,
    *,
    minimum_train_size: int = 1,
    minimum_test_size: int = 1,
) -> int:
    if length <= 0:
        raise ValueError("length must be positive")

    if isinstance(test_size, float):
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        split_idx = int(length * (1 - test_size))
    else:
        if not 0 < test_size < length:
            raise ValueError("Integer test_size must be between 1 and length - 1")
        split_idx = length - test_size

    if split_idx < minimum_train_size:
        raise ValueError("Training split is too small for the requested configuration.")
    if length - split_idx < minimum_test_size:
        raise ValueError("Test split is too small for the requested configuration.")
    return split_idx


def train_test_split_series(series: pd.Series, test_size: float | int = 0.2) -> Tuple[pd.Series, pd.Series]:
    split_idx = resolve_split_index(len(series), test_size)
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]
    return train, test


def train_val_test_split_series(
    series: pd.Series,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    if min(train_size, val_size, test_size) <= 0:
        raise ValueError("train_size, val_size and test_size must all be positive.")
    if not abs((train_size + val_size + test_size) - 1.0) < 1e-8:
        raise ValueError("train_size + val_size + test_size must equal 1")

    n = len(series)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)
    train = series.iloc[:train_end]
    val = series.iloc[train_end:val_end]
    test = series.iloc[val_end:]

    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("Series is too short for the requested train/val/test split.")
    return train, val, test
