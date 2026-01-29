from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..metrics import regression_report
from .lstm import create_supervised, _require_tensorflow


def build_gru_model(
    look_back: int,
    n_features: int = 1,
    units: int = 64,
    dense_units: int = 32,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
):
    _require_tensorflow()
    import tensorflow as tf

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(look_back, n_features)))
    model.add(tf.keras.layers.GRU(units, activation="relu"))
    model.add(tf.keras.layers.Dense(dense_units, activation="relu"))
    if dropout > 0:
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(1))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse")
    return model


@dataclass
class GruResult:
    model: object
    scaler: object
    metrics: dict
    predictions: pd.DataFrame
    history: object


def train_gru(
    series: pd.Series,
    look_back: int = 4,
    test_size: float = 0.2,
    epochs: int = 50,
    batch_size: int = 16,
    units: int = 64,
    dense_units: int = 32,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    verbose: int = 0,
) -> GruResult:
    _require_tensorflow()
    from sklearn.preprocessing import MinMaxScaler

    values = series.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    X, y = create_supervised(scaled, look_back=look_back)
    split_idx = int(len(X) * (1 - test_size))
    trainX, testX = X[:split_idx], X[split_idx:]
    trainY, testY = y[:split_idx], y[split_idx:]

    model = build_gru_model(
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

    return GruResult(model=model, scaler=scaler, metrics=metrics, predictions=preds, history=history)
