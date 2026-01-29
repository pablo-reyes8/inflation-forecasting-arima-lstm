from __future__ import annotations

from typing import Iterable

import numpy as np


def _to_numpy(values: Iterable) -> np.ndarray:
    return np.asarray(values, dtype=float)


def mse(y_true: Iterable, y_pred: Iterable) -> float:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: Iterable, y_pred: Iterable) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: Iterable, y_pred: Iterable) -> float:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: Iterable, y_pred: Iterable) -> float:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def regression_report(y_true: Iterable, y_pred: Iterable) -> dict:
    return {
        "r2": r2(y_true, y_pred),
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
    }
