from __future__ import annotations

import pandas as pd


def make_lag_features(series: pd.Series, lags: int = 4, dropna: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    if lags < 1:
        raise ValueError("lags must be >= 1")

    df = pd.DataFrame({"y": series.astype(float)})
    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = series.shift(lag)

    if dropna:
        df = df.dropna()
    if df.empty:
        raise ValueError("Lag feature construction produced an empty dataset. Increase series length or reduce lags.")

    y = df["y"]
    X = df.drop(columns=["y"])
    return X, y
