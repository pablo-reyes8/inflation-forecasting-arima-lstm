from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Optional

import pandas as pd

from ...datasets.splits import resolve_split_index
from ...metrics import regression_report


@dataclass
class ArimaResult:
    order: tuple[int, int, int]
    model_fit: object
    metrics: dict
    predictions: pd.Series


def _require_statsmodels():
    try:
        from statsmodels.tsa.arima.model import ARIMA  # noqa: F401
    except ImportError as exc:
        raise ImportError("statsmodels is required for ARIMA. Install with `pip install statsmodels`.") from exc


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


def fit_arima(series: pd.Series, order: tuple[int, int, int]) -> object:
    _require_statsmodels()
    from statsmodels.tsa.arima.model import ARIMA

    series = _coerce_supported_index(series)
    model = ARIMA(series, order=order)
    return model.fit()


def forecast_arima(model_fit: object, steps: int) -> pd.Series:
    forecast = model_fit.forecast(steps=steps)
    return forecast


def evaluate_arima(series: pd.Series, order: tuple[int, int, int], test_size: float = 0.2) -> ArimaResult:
    split_idx = resolve_split_index(len(series), test_size, minimum_train_size=max(8, sum(order) + 2))
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]

    model_fit = fit_arima(train, order=order)
    preds = model_fit.forecast(steps=len(test))
    metrics = regression_report(test.values, preds.values)
    metrics.update({"aic": model_fit.aic, "bic": model_fit.bic, "train_rows": int(len(train)), "test_rows": int(len(test))})
    predictions = pd.Series(preds.values, index=test.index, name="prediction")
    return ArimaResult(order=order, model_fit=model_fit, metrics=metrics, predictions=predictions)


def grid_search_arima(
    series: pd.Series,
    p_range: Iterable[int] = (0, 1, 2, 3),
    d_range: Iterable[int] = (0, 1),
    q_range: Iterable[int] = (0, 1, 2, 3),
    test_size: float = 0.2,
    max_models: Optional[int] = None,
) -> list[ArimaResult]:
    _require_statsmodels()

    results: list[ArimaResult] = []
    combos = list(product(p_range, d_range, q_range))
    if max_models:
        combos = combos[:max_models]

    for order in combos:
        try:
            res = evaluate_arima(series, order=order, test_size=test_size)
        except Exception:
            continue
        results.append(res)

    results.sort(key=lambda r: r.metrics["rmse"])
    return results
