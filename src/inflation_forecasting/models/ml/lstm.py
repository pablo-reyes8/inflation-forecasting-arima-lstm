from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ...metrics import regression_report
from .common import (
    _require_keras_tuner,
    _require_tensorflow,
    build_prediction_frame,
    create_supervised,
    history_to_frame,
    infer_future_index,
    inverse_transform_1d,
    prepare_sequence_split,
    set_random_seed,
)


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
    model.add(tf.keras.layers.LSTM(units, activation="tanh"))
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
    history_frame: pd.DataFrame


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
    validation_split: float = 0.1,
    patience: int = 10,
    random_seed: int = 42,
    verbose: int = 0,
) -> LstmResult:
    _require_tensorflow()
    import tensorflow as tf

    split = prepare_sequence_split(series, look_back=look_back, test_size=test_size)
    set_random_seed(random_seed)

    model = build_lstm_model(
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

    return LstmResult(
        model=model,
        scaler=split.scaler,
        metrics=metrics,
        predictions=predictions,
        history=history,
        history_frame=history_frame,
    )


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
    epochs: int = 50,
    random_seed: int = 42,
) -> LstmTuningResult:
    _require_tensorflow()
    _require_keras_tuner()

    import keras_tuner as kt

    split = prepare_sequence_split(series, look_back=look_back, test_size=test_size)
    set_random_seed(random_seed)

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
    tuner.search(split.trainX, split.trainY, validation_split=0.1, epochs=epochs, shuffle=False, verbose=0)

    best_model = tuner.get_best_models(num_models=1)[0]
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0].values
    return LstmTuningResult(best_model=best_model, best_hyperparameters=best_hp, tuner=tuner)


def forecast_future(model: object, scaler: object, series: pd.Series, look_back: int, steps: int) -> pd.Series:
    _require_tensorflow()
    values = series.astype(float).to_numpy().reshape(-1, 1)
    scaled = scaler.transform(values)

    window = scaled[-look_back:].reshape(1, look_back, 1)
    preds = []
    for _ in range(steps):
        pred = model.predict(window, verbose=0)[0][0]
        preds.append(pred)
        window = np.concatenate([window[:, 1:, :], np.array([[[pred]]])], axis=1)

    preds_inv = inverse_transform_1d(scaler, np.asarray(preds))
    future_index = infer_future_index(series, steps)
    return pd.Series(preds_inv, index=future_index, name="forecast")


def save_lstm_artifacts(model: object, scaler: object, model_path: str, scaler_path: str) -> None:
    _require_tensorflow()
    try:
        import joblib
    except ImportError as exc:
        raise ImportError("joblib is required to save scalers. Install with `pip install joblib`.") from exc

    model.save(model_path)
    joblib.dump(scaler, scaler_path)


def load_lstm_artifacts(model_path: str, scaler_path: str) -> tuple[object, object]:
    _require_tensorflow()
    try:
        import joblib
    except ImportError as exc:
        raise ImportError("joblib is required to load scalers. Install with `pip install joblib`.") from exc
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler
