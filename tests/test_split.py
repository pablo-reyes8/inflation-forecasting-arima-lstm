import pandas as pd

from inflation_forecasting.split import train_test_split_series, train_val_test_split_series


def test_train_test_split_series():
    series = pd.Series(range(10))
    train, test = train_test_split_series(series, test_size=0.2)
    assert len(train) == 8
    assert len(test) == 2


def test_train_val_test_split_series():
    series = pd.Series(range(20))
    train, val, test = train_val_test_split_series(series, train_size=0.7, val_size=0.15, test_size=0.15)
    assert len(train) == 14
    assert len(val) == 3
    assert len(test) == 3
