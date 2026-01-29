from __future__ import annotations

from typing import Tuple

import pandas as pd


def make_lag_features(series: pd.Series, lags: int = 4, dropna: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.DataFrame({"y": series})
    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = series.shift(lag)
    if dropna:
        df = df.dropna()
    y = df["y"]
    X = df.drop(columns=["y"])
    return X, y
