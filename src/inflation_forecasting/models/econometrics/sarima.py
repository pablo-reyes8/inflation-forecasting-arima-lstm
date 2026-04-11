from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ...datasets.splits import resolve_split_index
from ...metrics import regression_report


def _require_statsmodels():
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX  # noqa: F401
    except ImportError as exc:
        raise ImportError("statsmodels is required for SARIMA. Install with `pip install statsmodels`.") from exc


def _coerce_supported_index(series: pd.Series) -> pd.Series:
    if isinstance(series.index, pd.DatetimeIndex):
        months = set(series.index.month)
        if months.issubset({3, 6, 9, 12}):
            converted = series.copy()
            converted.index = series.index.to_period("Q")
            return converted
        if len(months) >= 6:
            converted = series.copy()
            converted.index = series.index.to_period("M")
            return converted
    return series


def infer_seasonal_period(series: pd.Series) -> Optional[int]:
    index = series.index
    freq = None
    if hasattr(index, "freq") and index.freq is not None:
        freq = index.freqstr
    if freq is None:
        freq = pd.infer_freq(index)

    if isinstance(index, pd.DatetimeIndex) and not index.empty and not freq:
        months = set(index.month)
        if months.issubset({3, 6, 9, 12}):
            return 4
        if len(months) >= 6:
            return 12

    if not freq:
        return None

    freq = freq.upper()
    if freq.startswith("Q"):
        return 4
    if freq.startswith("M"):
        return 12
    if freq.startswith("W"):
        return 52
    if freq.startswith("D"):
        return 7

    return None


@dataclass
class SarimaResult:
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    model_fit: object
    metrics: dict
    predictions: pd.Series


def fit_sarima(
    series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> object:
    _require_statsmodels()
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    series = _coerce_supported_index(series)
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(disp=False)


def evaluate_sarima(
    series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    test_size: float = 0.2,
) -> SarimaResult:
    split_idx = resolve_split_index(len(series), test_size, minimum_train_size=max(8, sum(order) + sum(seasonal_order[:3]) + 2))
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]

    model_fit = fit_sarima(train, order=order, seasonal_order=seasonal_order)
    preds = model_fit.forecast(steps=len(test))
    metrics = regression_report(test.values, preds.values)
    metrics.update({"aic": model_fit.aic, "bic": model_fit.bic, "train_rows": int(len(train)), "test_rows": int(len(test))})
    predictions = pd.Series(preds.values, index=test.index, name="prediction")
    return SarimaResult(
        order=order,
        seasonal_order=seasonal_order,
        model_fit=model_fit,
        metrics=metrics,
        predictions=predictions,
    )
