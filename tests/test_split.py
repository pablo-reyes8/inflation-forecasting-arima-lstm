import pandas as pd
import pytest

from inflation_forecasting.datasets.splits import train_test_split_series, train_val_test_split_series


def test_train_test_split_series():
    series = pd.Series(range(10))
    train, test = train_test_split_series(series, test_size=0.2)
    assert len(train) == 8
    assert len(test) == 2


def test_train_test_split_series_supports_integer_test_size():
    series = pd.Series(range(10))
    train, test = train_test_split_series(series, test_size=3)
    assert len(train) == 7
    assert len(test) == 3


def test_train_val_test_split_series():
    series = pd.Series(range(20))
    train, val, test = train_val_test_split_series(series, train_size=0.7, val_size=0.15, test_size=0.15)
    assert len(train) == 14
    assert len(val) == 3
    assert len(test) == 3


def test_train_val_test_split_series_rejects_invalid_weights():
    with pytest.raises(ValueError):
        train_val_test_split_series(pd.Series(range(10)), train_size=0.5, val_size=0.4, test_size=0.4)
