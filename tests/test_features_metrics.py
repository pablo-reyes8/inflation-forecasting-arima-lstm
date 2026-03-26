import numpy as np
import pandas as pd
import pytest

from inflation_forecasting.features import make_lag_features
from inflation_forecasting.metrics import mae, mape, mse, regression_report, rmse, r2, smape


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
    assert mape(y_true, y_pred) >= 0
    assert smape(y_true, y_pred) >= 0
    assert r2(y_true, y_pred) < 1


def test_make_lag_features_rejects_invalid_lag_count():
    with pytest.raises(ValueError):
        make_lag_features(pd.Series([1, 2, 3]), lags=0)


def test_regression_report_contains_extended_metrics():
    report = regression_report([1.0, 2.0], [1.0, 2.5])
    assert "mape" in report
    assert "smape" in report
