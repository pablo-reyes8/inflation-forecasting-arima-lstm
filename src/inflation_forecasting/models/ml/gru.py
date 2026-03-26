from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ...metrics import regression_report
from .common import (
    _require_tensorflow,
    build_prediction_frame,
    history_to_frame,
    inverse_transform_1d,
    prepare_sequence_split,
    set_random_seed,
)


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
    model.add(tf.keras.layers.GRU(units, activation="tanh"))
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
    history_frame: pd.DataFrame


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
    validation_split: float = 0.1,
    patience: int = 10,
    random_seed: int = 42,
    verbose: int = 0,
) -> GruResult:
    _require_tensorflow()
    import tensorflow as tf

    split = prepare_sequence_split(series, look_back=look_back, test_size=test_size)
    set_random_seed(random_seed)

    model = build_gru_model(
        look_back=look_back,
        n_features=1,
        units=units,
        dense_units=dense_units,
        dropout=dropout,
        learning_rate=learning_rate,
    )

    callbacks = []
    effective_validation_split = validation_split if len(split.trainX) >= 10 else 0.0
    if effective_validation_split > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
            )
        )

    history = model.fit(
        split.trainX,
        split.trainY,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=effective_validation_split,
        callbacks=callbacks,
        shuffle=False,
        verbose=verbose,
    )

    train_pred = model.predict(split.trainX, verbose=0)
    test_pred = model.predict(split.testX, verbose=0)

    train_pred_inv = inverse_transform_1d(split.scaler, train_pred)
    test_pred_inv = inverse_transform_1d(split.scaler, test_pred)
    trainY_inv = inverse_transform_1d(split.scaler, split.trainY)
    testY_inv = inverse_transform_1d(split.scaler, split.testY)

    metrics = regression_report(testY_inv, test_pred_inv)
    metrics.update(
        {
            "train_windows": int(len(split.train_index)),
            "test_windows": int(len(split.test_index)),
            "look_back": int(look_back),
        }
    )

    predictions = build_prediction_frame(
        train_true=trainY_inv,
        train_pred=train_pred_inv,
        train_index=split.train_index,
        test_true=testY_inv,
        test_pred=test_pred_inv,
        test_index=split.test_index,
    )
    history_frame = history_to_frame(history)

    return GruResult(
        model=model,
        scaler=split.scaler,
        metrics=metrics,
        predictions=predictions,
        history=history,
        history_frame=history_frame,
    )
