from __future__ import annotations

from typing import Iterable

import numpy as np


def _to_numpy(values: Iterable) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must share the same shape, got {y_true.shape} and {y_pred.shape}.")
    if y_true.size == 0:
        raise ValueError("Metrics require at least one observation.")


def mse(y_true: Iterable, y_pred: Iterable) -> float:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    _validate_inputs(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: Iterable, y_pred: Iterable) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: Iterable, y_pred: Iterable) -> float:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    _validate_inputs(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: Iterable, y_pred: Iterable) -> float:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    _validate_inputs(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def mape(y_true: Iterable, y_pred: Iterable) -> float:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    _validate_inputs(y_true, y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true: Iterable, y_pred: Iterable) -> float:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    _validate_inputs(y_true, y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]) * 100)


def regression_report(y_true: Iterable, y_pred: Iterable) -> dict:
    return {
        "r2": r2(y_true, y_pred),
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }
