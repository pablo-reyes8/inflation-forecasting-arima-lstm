from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ...datasets.splits import resolve_split_index


def _require_tensorflow():
    try:
        import tensorflow as tf  # noqa: F401
    except ImportError as exc:
        raise ImportError("tensorflow is required for LSTM/GRU models. Install with `pip install tensorflow`.") from exc


def _require_keras_tuner():
    try:
        import keras_tuner  # noqa: F401
    except ImportError as exc:
        raise ImportError("keras-tuner is required for hyperparameter tuning. Install with `pip install keras-tuner`.") from exc


def set_random_seed(seed: int) -> None:
    _require_tensorflow()
    import tensorflow as tf

    np.random.seed(seed)
    tf.random.set_seed(seed)


def create_supervised(values: np.ndarray, look_back: int) -> tuple[np.ndarray, np.ndarray]:
    if look_back < 1:
        raise ValueError("look_back must be >= 1")
    if len(values) <= look_back:
        raise ValueError("Sequence is too short for the requested look_back window.")

    X, y = [], []
    for index in range(len(values) - look_back):
        X.append(values[index : index + look_back])
        y.append(values[index + look_back])
    return np.array(X), np.array(y)


@dataclass
class SequenceSplit:
    scaler: object
    trainX: np.ndarray
    testX: np.ndarray
    trainY: np.ndarray
    testY: np.ndarray
    train_index: pd.Index
    test_index: pd.Index
    split_idx: int


def prepare_sequence_split(series: pd.Series, look_back: int, test_size: float = 0.2) -> SequenceSplit:
    from sklearn.preprocessing import MinMaxScaler

    numeric = series.astype(float)
    if numeric.isna().any():
        raise ValueError("Sequence models require a series without missing values. Interpolate or impute first.")

    split_idx = resolve_split_index(
        len(numeric),
        test_size,
        minimum_train_size=look_back + 1,
        minimum_test_size=1,
    )

    train_values = numeric.iloc[:split_idx].to_numpy().reshape(-1, 1)
    evaluation_values = numeric.iloc[split_idx - look_back :].to_numpy().reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_values)
    evaluation_scaled = scaler.transform(evaluation_values)

    trainX, trainY = create_supervised(train_scaled, look_back)
    testX, testY = create_supervised(evaluation_scaled, look_back)

    train_index = numeric.index[look_back:split_idx]
    test_index = numeric.index[split_idx:]
    return SequenceSplit(
        scaler=scaler,
        trainX=trainX,
        testX=testX,
        trainY=trainY,
        testY=testY,
        train_index=train_index,
        test_index=test_index,
        split_idx=split_idx,
    )


def inverse_transform_1d(scaler: object, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values).reshape(-1, 1)
    return scaler.inverse_transform(array).flatten()


def history_to_frame(history: object) -> pd.DataFrame:
    if hasattr(history, "history"):
        return pd.DataFrame(history.history)
    return pd.DataFrame()


def build_prediction_frame(
    *,
    train_true: np.ndarray,
    train_pred: np.ndarray,
    train_index: pd.Index,
    test_true: np.ndarray,
    test_pred: np.ndarray,
    test_index: pd.Index,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y_true": np.concatenate([train_true, test_true]),
            "y_pred": np.concatenate([train_pred, test_pred]),
            "split": ["train"] * len(train_index) + ["test"] * len(test_index),
        },
        index=train_index.append(test_index),
    )


def infer_future_index(series: pd.Series, steps: int) -> pd.Index:
    inferred_frequency = getattr(series.index, "freqstr", None)
    if inferred_frequency is None and len(series.index) >= 3:
        inferred_frequency = pd.infer_freq(series.index)
    if inferred_frequency is None:
        inferred_frequency = "QE-DEC"
    return pd.date_range(start=series.index[-1], periods=steps + 1, freq=inferred_frequency)[1:]
