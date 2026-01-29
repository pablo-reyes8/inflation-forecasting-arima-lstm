import numpy as np
import pandas as pd

from inflation_forecasting.features import make_lag_features
from inflation_forecasting.metrics import mae, mse, rmse, r2


def test_make_lag_features():
    series = pd.Series([1, 2, 3, 4, 5])
    X, y = make_lag_features(series, lags=2)
    assert list(X.columns) == ["lag_1", "lag_2"]
    assert len(X) == len(y)
    assert len(X) == 3


def test_metrics_basic():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.5, 2.5])
    assert mse(y_true, y_pred) > 0
    assert rmse(y_true, y_pred) > 0
    assert mae(y_true, y_pred) > 0
    assert r2(y_true, y_pred) < 1
