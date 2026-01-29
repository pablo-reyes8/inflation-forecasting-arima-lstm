from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ...metrics import regression_report


def _require_statsmodels():
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX  # noqa: F401
    except ImportError as exc:
        raise ImportError("statsmodels is required for ARIMAX. Install with `pip install statsmodels`.") from exc


@dataclass
class ArimaxResult:
    order: tuple[int, int, int]
    model_fit: object
    metrics: dict
    predictions: pd.Series


def _align_exog(series: pd.Series, exog: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    exog = exog.copy()
    exog = exog.loc[series.index]
    aligned = pd.concat([series, exog], axis=1).dropna()
    series_aligned = aligned.iloc[:, 0]
    exog_aligned = aligned.iloc[:, 1:]
    return series_aligned, exog_aligned


def fit_arimax(series: pd.Series, exog: pd.DataFrame, order: tuple[int, int, int]) -> object:
    _require_statsmodels()
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(series, order=order, exog=exog, enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(disp=False)


def evaluate_arimax(
    series: pd.Series,
    exog: pd.DataFrame,
    order: tuple[int, int, int],
    test_size: float = 0.2,
) -> ArimaxResult:
    series, exog = _align_exog(series, exog)
    n = len(series)
    split_idx = int(n * (1 - test_size))
    train_y, test_y = series.iloc[:split_idx], series.iloc[split_idx:]
    train_x, test_x = exog.iloc[:split_idx], exog.iloc[split_idx:]

    model_fit = fit_arimax(train_y, train_x, order=order)
    preds = model_fit.forecast(steps=len(test_y), exog=test_x)
    metrics = regression_report(test_y.values, preds.values)
    predictions = pd.Series(preds.values, index=test_y.index, name="prediction")
    return ArimaxResult(order=order, model_fit=model_fit, metrics=metrics, predictions=predictions)
