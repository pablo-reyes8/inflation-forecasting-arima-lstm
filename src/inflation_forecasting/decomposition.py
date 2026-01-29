from __future__ import annotations

import pandas as pd


def hp_filter(series: pd.Series, lamb: float = 1600.0) -> tuple[pd.Series, pd.Series]:
    try:
        from statsmodels.tsa.filters.hp_filter import hpfilter
    except ImportError as exc:
        raise ImportError("statsmodels is required for HP filter. Install with `pip install statsmodels`.") from exc
    cycle, trend = hpfilter(series, lamb=lamb)
    return trend, cycle


def seasonal_decompose_series(
    series: pd.Series,
    model: str = "additive",
    period: int | None = None,
):
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
    except ImportError as exc:
        raise ImportError("statsmodels is required for seasonal decomposition. Install with `pip install statsmodels`.") from exc
    return seasonal_decompose(series, model=model, period=period)
