"""Compatibility wrapper for `inflation_forecasting.datasets.splits`."""

from ..datasets.splits import resolve_split_index, train_test_split_series, train_val_test_split_series

__all__ = ["resolve_split_index", "train_test_split_series", "train_val_test_split_series"]
