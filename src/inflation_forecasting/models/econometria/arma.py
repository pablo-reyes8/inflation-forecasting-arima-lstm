from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Optional

import pandas as pd

from ...datasets.splits import resolve_split_index
from ...metrics import regression_report
from .arima import fit_arima


@dataclass
class ArmaResult:
    order: tuple[int, int]
    model_fit: object
    metrics: dict
    predictions: pd.Series


def evaluate_arma(series: pd.Series, order: tuple[int, int], test_size: float = 0.2) -> ArmaResult:
    p, q = order
    split_idx = resolve_split_index(len(series), test_size, minimum_train_size=max(8, p + q + 2))
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]

    model_fit = fit_arima(train, order=(p, 0, q))
    preds = model_fit.forecast(steps=len(test))
    metrics = regression_report(test.values, preds.values)
    metrics.update({"aic": model_fit.aic, "bic": model_fit.bic, "train_rows": int(len(train)), "test_rows": int(len(test))})
    predictions = pd.Series(preds.values, index=test.index, name="prediction")
    return ArmaResult(order=order, model_fit=model_fit, metrics=metrics, predictions=predictions)


def grid_search_arma(
    series: pd.Series,
    p_range: Iterable[int] = (0, 1, 2, 3),
    q_range: Iterable[int] = (0, 1, 2, 3),
    test_size: float = 0.2,
    max_models: Optional[int] = None,
) -> list[ArmaResult]:
    results: list[ArmaResult] = []
    combos = list(product(p_range, q_range))
    if max_models:
        combos = combos[:max_models]

    for order in combos:
        try:
            res = evaluate_arma(series, order=order, test_size=test_size)
        except Exception:
            continue
        results.append(res)

    results.sort(key=lambda r: r.metrics["rmse"])
    return results
