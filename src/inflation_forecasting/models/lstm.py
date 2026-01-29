from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..metrics import regression_report


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


def create_supervised(values: np.ndarray, look_back: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(values) - look_back):
        X.append(values[i : i + look_back])
        y.append(values[i + look_back])
    return np.array(X), np.array(y)


def build_lstm_model(
    look_back: int,
    n_features: int = 1,
    units: int = 100,
    dense_units: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
):
    _require_tensorflow()
    import tensorflow as tf

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(look_back, n_features)))
    model.add(tf.keras.layers.LSTM(units, activation="relu"))
    model.add(tf.keras.layers.Dense(dense_units, activation="relu"))
    if dropout > 0:
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(1))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse")
    return model


@dataclass
class LstmResult:
    model: object
    scaler: object
    metrics: dict
    predictions: pd.DataFrame
    history: object


def train_lstm(
    series: pd.Series,
    look_back: int = 4,
    test_size: float = 0.2,
    epochs: int = 50,
    batch_size: int = 16,
    units: int = 100,
    dense_units: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    verbose: int = 0,
) -> LstmResult:
    _require_tensorflow()
    from sklearn.preprocessing import MinMaxScaler

    values = series.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    X, y = create_supervised(scaled, look_back=look_back)
    split_idx = int(len(X) * (1 - test_size))
    trainX, testX = X[:split_idx], X[split_idx:]
    trainY, testY = y[:split_idx], y[split_idx:]

    model = build_lstm_model(
        look_back=look_back,
        n_features=1,
        units=units,
        dense_units=dense_units,
        dropout=dropout,
        learning_rate=learning_rate,
    )

    history = model.fit(
        trainX,
        trainY,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(testX, testY),
        verbose=verbose,
    )

    train_pred = model.predict(trainX, verbose=0)
    test_pred = model.predict(testX, verbose=0)

    train_pred_inv = scaler.inverse_transform(train_pred)
    test_pred_inv = scaler.inverse_transform(test_pred)
    trainY_inv = scaler.inverse_transform(trainY)
    testY_inv = scaler.inverse_transform(testY)

    metrics = regression_report(testY_inv.flatten(), test_pred_inv.flatten())

    train_index = series.index[look_back:look_back + len(train_pred_inv)]
    test_index = series.index[look_back + len(train_pred_inv) : look_back + len(train_pred_inv) + len(test_pred_inv)]

    preds = pd.DataFrame(
        {
            "y_true": np.concatenate([trainY_inv.flatten(), testY_inv.flatten()]),
            "y_pred": np.concatenate([train_pred_inv.flatten(), test_pred_inv.flatten()]),
            "split": ["train"] * len(train_pred_inv) + ["test"] * len(test_pred_inv),
        },
        index=train_index.append(test_index),
    )

    return LstmResult(model=model, scaler=scaler, metrics=metrics, predictions=preds, history=history)


@dataclass
class LstmTuningResult:
    best_model: object
    best_hyperparameters: dict
    tuner: object


def tune_lstm(
    series: pd.Series,
    look_back: int = 4,
    test_size: float = 0.2,
    max_trials: int = 10,
    executions_per_trial: int = 2,
    directory: str = "tuning",
    project_name: str = "lstm",
) -> LstmTuningResult:
    _require_tensorflow()
    _require_keras_tuner()

    import keras_tuner as kt
    from sklearn.preprocessing import MinMaxScaler

    values = series.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    X, y = create_supervised(scaled, look_back=look_back)
    split_idx = int(len(X) * (1 - test_size))
    trainX, testX = X[:split_idx], X[split_idx:]
    trainY, testY = y[:split_idx], y[split_idx:]

    def model_builder(hp):
        units = hp.Int("units", min_value=32, max_value=256, step=32)
        dense_units = hp.Int("dense_units", min_value=16, max_value=128, step=16)
        dropout = hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)
        lr = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
        return build_lstm_model(
            look_back=look_back,
            n_features=1,
            units=units,
            dense_units=dense_units,
            dropout=dropout,
            learning_rate=lr,
        )

    tuner = kt.RandomSearch(
        model_builder,
        objective="val_loss",
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory=directory,
        project_name=project_name,
        overwrite=True,
    )

    tuner.search(trainX, trainY, validation_data=(testX, testY), epochs=50, verbose=0)
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0].values

    return LstmTuningResult(best_model=best_model, best_hyperparameters=best_hp, tuner=tuner)


def forecast_future(model: object, scaler: object, series: pd.Series, look_back: int, steps: int) -> pd.Series:
    _require_tensorflow()
    values = series.values.reshape(-1, 1)
    scaled = scaler.transform(values)

    window = scaled[-look_back:].reshape(1, look_back, 1)
    preds = []
    for _ in range(steps):
        pred = model.predict(window, verbose=0)[0][0]
        preds.append(pred)
        new_window = np.append(window[:, 1:, :], [[[pred]]], axis=1)
        window = new_window

    preds = np.array(preds).reshape(-1, 1)
    preds_inv = scaler.inverse_transform(preds).flatten()
    future_index = pd.date_range(series.index[-1], periods=steps + 1, freq="Q")[1:]
    return pd.Series(preds_inv, index=future_index, name="forecast")
