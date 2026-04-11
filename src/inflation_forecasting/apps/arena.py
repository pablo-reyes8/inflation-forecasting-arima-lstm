from __future__ import annotations

from dataclasses import dataclass, field
from importlib.util import find_spec
from time import perf_counter
from typing import Any, Literal

import pandas as pd

from ..datasets.features import make_lag_features
from ..datasets.io import read_tabular_data as read_uploaded_tabular_data
from ..datasets.preprocessing import add_quarterly_date, validate_required_columns
from ..datasets.splits import train_val_test_split_series
from ..metrics import regression_report
from ..models.econometrics.arima import fit_arima
from ..models.econometrics.arimax import fit_arimax
from ..models.econometrics.sarima import fit_sarima, infer_seasonal_period
from ..models.ml.common import create_supervised, set_random_seed
from ..models.ml.gru import build_gru_model
from ..models.ml.lstm import build_lstm_model


ModelKey = Literal[
    "arima",
    "arma",
    "sarima",
    "arimax",
    "prophet",
    "linear_regression",
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "lstm",
    "gru",
]


MODEL_CATALOG: dict[ModelKey, dict[str, Any]] = {
    "arima": {"label": "ARIMA", "family": "Econometrics", "requires": ("statsmodels",)},
    "arma": {"label": "ARMA", "family": "Econometrics", "requires": ("statsmodels",)},
    "sarima": {"label": "SARIMA", "family": "Econometrics", "requires": ("statsmodels",)},
    "arimax": {"label": "ARIMAX", "family": "Econometrics", "requires": ("statsmodels",), "needs_exog": True},
    "prophet": {"label": "Prophet", "family": "Econometrics", "requires": ("prophet",)},
    "linear_regression": {"label": "Linear Regression", "family": "Machine Learning", "requires": ("sklearn",)},
    "random_forest": {"label": "Random Forest", "family": "Machine Learning", "requires": ("sklearn",)},
    "gradient_boosting": {"label": "Gradient Boosting", "family": "Machine Learning", "requires": ("sklearn",)},
    "xgboost": {"label": "XGBoost", "family": "Machine Learning", "requires": ("sklearn", "xgboost")},
    "lstm": {"label": "LSTM", "family": "Deep Learning", "requires": ("tensorflow",)},
    "gru": {"label": "GRU", "family": "Deep Learning", "requires": ("tensorflow",)},
}


@dataclass(frozen=True)
class ArenaDataset:
    name: str
    frame: pd.DataFrame
    target_col: str
    exog_cols: tuple[str, ...] = ()
    entity_col: str | None = None
    entity_value: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def series(self) -> pd.Series:
        return self.frame[self.target_col]

    @property
    def exog(self) -> pd.DataFrame | None:
        if not self.exog_cols:
            return None
        return self.frame.loc[:, list(self.exog_cols)]


@dataclass(frozen=True)
class ArenaRunConfig:
    train_size: float = 0.6
    val_size: float = 0.2
    test_size: float = 0.2
    ranking_metric: str = "rmse"
    arima_order: tuple[int, int, int] = (1, 0, 1)
    arma_order: tuple[int, int] = (1, 1)
    sarima_order: tuple[int, int, int] = (1, 0, 1)
    sarima_seasonal_order: tuple[int, int, int, int] | None = None
    lags: int = 4
    look_back: int = 4
    epochs: int = 40
    batch_size: int = 16
    validation_split: float = 0.1
    patience: int = 8
    random_seed: int = 42
    lstm_units: int = 64
    lstm_dense_units: int = 32
    gru_units: int = 64
    gru_dense_units: int = 32
    dropout: float = 0.1
    learning_rate: float = 1e-3


@dataclass
class ArenaModelResult:
    model_key: ModelKey
    label: str
    family: str
    status: Literal["ok", "error", "skipped"]
    validation_metrics: dict[str, float] | None = None
    test_metrics: dict[str, float] | None = None
    predictions: pd.DataFrame | None = None
    duration_seconds: float | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    history: pd.DataFrame | None = None
    error: str | None = None
    notes: list[str] = field(default_factory=list)


def _dependency_available(module_name: str) -> bool:
    return find_spec(module_name) is not None


def available_model_catalog(*, has_exog: bool = False) -> dict[ModelKey, dict[str, Any]]:
    catalog: dict[ModelKey, dict[str, Any]] = {}
    for model_key, spec in MODEL_CATALOG.items():
        available = all(_dependency_available(module_name) for module_name in spec["requires"])
        if spec.get("needs_exog") and not has_exog:
            available = False
        catalog[model_key] = {
            **spec,
            "available": available,
            "reason": None if available else _unavailable_reason(spec, has_exog=has_exog),
        }
    return catalog


def _unavailable_reason(spec: dict[str, Any], *, has_exog: bool) -> str:
    if spec.get("needs_exog") and not has_exog:
        return "Needs exogenous regressors selected in the dataset panel."
    missing = [module_name for module_name in spec["requires"] if not _dependency_available(module_name)]
    return f"Missing optional dependency: {', '.join(missing)}."


def read_tabular_data(file_name: str, payload: bytes) -> pd.DataFrame:
    return read_uploaded_tabular_data(file_name, payload)


def _parse_date_index(
    df: pd.DataFrame,
    *,
    date_col: str | None = None,
    year_col: str | None = None,
    quarter_col: str | None = None,
) -> pd.DataFrame:
    if date_col:
        validate_required_columns(df, (date_col,))
        frame = df.copy()
        frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
        if frame[date_col].isna().any():
            raise ValueError(f"Column '{date_col}' contains invalid dates.")
        return frame.set_index(date_col).sort_index()
    if year_col and quarter_col:
        frame = add_quarterly_date(df, year_col=year_col, quarter_col=quarter_col, date_col="date")
        return frame.set_index("date").sort_index()
    raise ValueError("Select either a date column or a year/quarter pair.")


def prepare_arena_dataset(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    target_col: str,
    date_col: str | None = None,
    year_col: str | None = None,
    quarter_col: str | None = None,
    entity_col: str | None = None,
    entity_value: str | None = None,
    exog_cols: list[str] | None = None,
    interpolate_missing: bool = True,
) -> ArenaDataset:
    exog_cols = exog_cols or []
    columns_to_validate = [target_col, *exog_cols]
    if entity_col:
        columns_to_validate.append(entity_col)
    validate_required_columns(df, columns_to_validate)

    frame = df.copy()
    if entity_col and entity_value is not None:
        frame = frame.loc[frame[entity_col] == entity_value].copy()
    if frame.empty:
        raise ValueError("The selected dataset slice is empty.")

    frame = _parse_date_index(frame, date_col=date_col, year_col=year_col, quarter_col=quarter_col)
    selected_columns = [target_col, *exog_cols]
    frame = frame.loc[:, selected_columns]
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.sort_index()

    if interpolate_missing:
        frame = frame.interpolate(method="linear").ffill().bfill()

    if frame.isna().any().any():
        missing = [column for column in frame.columns if frame[column].isna().any()]
        raise ValueError(f"Missing values remain after preprocessing in columns: {missing}")

    if frame.index.duplicated().any():
        raise ValueError("The resulting time series has duplicate timestamps.")
    if len(frame) < 12:
        raise ValueError("The selected series is too short. Provide at least 12 observations.")

    inferred_frequency = None
    if len(frame.index) >= 3:
        inferred_frequency = pd.infer_freq(frame.index)

    metadata = {
        "rows": int(len(frame)),
        "start": frame.index.min(),
        "end": frame.index.max(),
        "frequency": inferred_frequency,
        "n_exog": int(len(exog_cols)),
    }
    return ArenaDataset(
        name=dataset_name,
        frame=frame,
        target_col=target_col,
        exog_cols=tuple(exog_cols),
        entity_col=entity_col,
        entity_value=entity_value,
        metadata=metadata,
    )


def parse_order_text(text: str, expected: int) -> tuple[int, ...]:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != expected:
        raise ValueError(f"Expected {expected} comma-separated values, got '{text}'.")
    return tuple(int(part) for part in parts)


def _infer_sarima_seasonal_order(series: pd.Series, config: ArenaRunConfig) -> tuple[int, int, int, int]:
    if config.sarima_seasonal_order is not None:
        return config.sarima_seasonal_order
    seasonal_period = infer_seasonal_period(series)
    if seasonal_period is None:
        return (0, 0, 0, 0)
    return (1, 0, 1, seasonal_period)


def _split_dataset(dataset: ArenaDataset, config: ArenaRunConfig) -> dict[str, pd.Series | pd.DataFrame | None]:
    train, val, test = train_val_test_split_series(
        dataset.series,
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
    )
    if len(train) < 8 or len(val) < 2 or len(test) < 2:
        raise ValueError("Series is too short for the selected train/validation/test split.")

    exog = dataset.exog
    if exog is None:
        return {"train": train, "val": val, "test": test, "train_exog": None, "val_exog": None, "test_exog": None}

    return {
        "train": train,
        "val": val,
        "test": test,
        "train_exog": exog.loc[train.index],
        "val_exog": exog.loc[val.index],
        "test_exog": exog.loc[test.index],
    }


def _prediction_frame(
    *,
    model_key: ModelKey,
    label: str,
    validation_true: pd.Series,
    validation_pred: pd.Series,
    test_true: pd.Series,
    test_pred: pd.Series,
) -> pd.DataFrame:
    validation = pd.DataFrame(
        {
            "actual": validation_true.values,
            "predicted": validation_pred.values,
            "phase": "validation",
            "model_key": model_key,
            "model": label,
        },
        index=validation_true.index,
    )
    test = pd.DataFrame(
        {
            "actual": test_true.values,
            "predicted": test_pred.values,
            "phase": "test",
            "model_key": model_key,
            "model": label,
        },
        index=test_true.index,
    )
    output = pd.concat([validation, test]).sort_index()
    output.index.name = "date"
    return output


def _fit_predict_prophet(train: pd.Series, future_index: pd.Index) -> pd.Series:
    from prophet import Prophet

    train_df = train.reset_index()
    train_df.columns = ["ds", "y"]
    model = Prophet()
    model.fit(train_df)
    future_df = pd.DataFrame({"ds": future_index})
    predictions = model.predict(future_df)["yhat"].to_numpy()
    return pd.Series(predictions, index=future_index, name="prediction")


def _instantiate_tabular_model(model_key: ModelKey, random_state: int) -> object:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    if model_key == "linear_regression":
        return LinearRegression()
    if model_key == "random_forest":
        return RandomForestRegressor(n_estimators=300, random_state=random_state)
    if model_key == "gradient_boosting":
        return GradientBoostingRegressor(random_state=random_state)
    if model_key == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
        )
    raise ValueError(f"Unsupported tabular model '{model_key}'.")


def _fit_predict_tabular(
    *,
    model_key: ModelKey,
    train: pd.Series,
    evaluate: pd.Series,
    lags: int,
    random_state: int,
) -> tuple[pd.Series, dict[str, Any]]:
    combined = pd.concat([train, evaluate])
    X_all, y_all = make_lag_features(combined, lags=lags)
    if not set(evaluate.index).issubset(set(X_all.index)):
        raise ValueError("Not enough history to build lag features for the evaluation window.")

    train_mask = X_all.index <= train.index[-1]
    X_train, y_train = X_all.loc[train_mask], y_all.loc[train_mask]
    X_eval, y_eval = X_all.loc[evaluate.index], y_all.loc[evaluate.index]

    model = _instantiate_tabular_model(model_key, random_state=random_state)
    model.fit(X_train, y_train)
    preds = model.predict(X_eval)
    return pd.Series(preds, index=y_eval.index, name="prediction"), {
        "train_rows": int(len(X_train)),
        "eval_rows": int(len(X_eval)),
        "lags": int(lags),
    }


def _fit_predict_sequence(
    *,
    builder: Any,
    train: pd.Series,
    evaluate: pd.Series,
    look_back: int,
    epochs: int,
    batch_size: int,
    validation_split: float,
    patience: int,
    random_seed: int,
    builder_params: dict[str, Any],
) -> tuple[pd.Series, pd.DataFrame, dict[str, Any]]:
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler

    if len(train) <= look_back:
        raise ValueError("Training window is too short for the selected look_back.")
    set_random_seed(random_seed)

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_values = train.astype(float).to_numpy().reshape(-1, 1)
    train_scaled = scaler.fit_transform(train_values)
    trainX, trainY = create_supervised(train_scaled, look_back)

    eval_context = pd.concat([train.tail(look_back), evaluate]).astype(float).to_numpy().reshape(-1, 1)
    eval_scaled = scaler.transform(eval_context)
    evalX, _ = create_supervised(eval_scaled, look_back)
    if len(evalX) != len(evaluate):
        raise ValueError("Could not align sequence windows with the evaluation horizon.")

    model = builder(look_back=look_back, n_features=1, **builder_params)
    callbacks = []
    effective_validation_split = validation_split if len(trainX) >= 10 else 0.0
    if effective_validation_split > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
            )
        )

    history = model.fit(
        trainX,
        trainY,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=effective_validation_split,
        callbacks=callbacks,
        shuffle=False,
        verbose=0,
    )
    preds = model.predict(evalX, verbose=0)
    predictions = scaler.inverse_transform(preds).flatten()
    history_frame = pd.DataFrame(history.history)
    return pd.Series(predictions, index=evaluate.index, name="prediction"), history_frame, {
        "train_windows": int(len(trainX)),
        "eval_windows": int(len(evalX)),
        "look_back": int(look_back),
    }


def _evaluate_arima_like(dataset_parts: dict[str, Any], config: ArenaRunConfig, model_key: ModelKey) -> ArenaModelResult:
    train = dataset_parts["train"]
    val = dataset_parts["val"]
    test = dataset_parts["test"]

    if model_key == "arima":
        fit_val = fit_arima(train, order=config.arima_order)
        val_pred = pd.Series(fit_val.forecast(steps=len(val)).to_numpy(), index=val.index, name="prediction")
        fit_test = fit_arima(pd.concat([train, val]), order=config.arima_order)
        test_pred = pd.Series(fit_test.forecast(steps=len(test)).to_numpy(), index=test.index, name="prediction")
        parameters = {"order": config.arima_order}
    elif model_key == "arma":
        order = (config.arma_order[0], 0, config.arma_order[1])
        fit_val = fit_arima(train, order=order)
        val_pred = pd.Series(fit_val.forecast(steps=len(val)).to_numpy(), index=val.index, name="prediction")
        fit_test = fit_arima(pd.concat([train, val]), order=order)
        test_pred = pd.Series(fit_test.forecast(steps=len(test)).to_numpy(), index=test.index, name="prediction")
        parameters = {"order": config.arma_order}
    else:
        seasonal_order = _infer_sarima_seasonal_order(pd.concat([train, val, test]), config)
        fit_val = fit_sarima(train, order=config.sarima_order, seasonal_order=seasonal_order)
        val_pred = pd.Series(fit_val.forecast(steps=len(val)).to_numpy(), index=val.index, name="prediction")
        fit_test = fit_sarima(pd.concat([train, val]), order=config.sarima_order, seasonal_order=seasonal_order)
        test_pred = pd.Series(fit_test.forecast(steps=len(test)).to_numpy(), index=test.index, name="prediction")
        parameters = {"order": config.sarima_order, "seasonal_order": seasonal_order}

    validation_metrics = regression_report(val.values, val_pred.values)
    test_metrics = regression_report(test.values, test_pred.values)
    return ArenaModelResult(
        model_key=model_key,
        label=MODEL_CATALOG[model_key]["label"],
        family=MODEL_CATALOG[model_key]["family"],
        status="ok",
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        predictions=_prediction_frame(
            model_key=model_key,
            label=MODEL_CATALOG[model_key]["label"],
            validation_true=val,
            validation_pred=val_pred,
            test_true=test,
            test_pred=test_pred,
        ),
        parameters=parameters,
    )


def _evaluate_arimax(dataset_parts: dict[str, Any], config: ArenaRunConfig) -> ArenaModelResult:
    train = dataset_parts["train"]
    val = dataset_parts["val"]
    test = dataset_parts["test"]
    train_exog = dataset_parts["train_exog"]
    val_exog = dataset_parts["val_exog"]
    test_exog = dataset_parts["test_exog"]
    if train_exog is None or val_exog is None or test_exog is None:
        raise ValueError("ARIMAX requires exogenous regressors.")

    fit_val = fit_arimax(train, train_exog, order=config.arima_order)
    val_pred = pd.Series(fit_val.forecast(steps=len(val), exog=val_exog).to_numpy(), index=val.index, name="prediction")
    train_val = pd.concat([train, val])
    train_val_exog = pd.concat([train_exog, val_exog])
    fit_test = fit_arimax(train_val, train_val_exog, order=config.arima_order)
    test_pred = pd.Series(
        fit_test.forecast(steps=len(test), exog=test_exog).to_numpy(),
        index=test.index,
        name="prediction",
    )
    validation_metrics = regression_report(val.values, val_pred.values)
    test_metrics = regression_report(test.values, test_pred.values)
    return ArenaModelResult(
        model_key="arimax",
        label=MODEL_CATALOG["arimax"]["label"],
        family=MODEL_CATALOG["arimax"]["family"],
        status="ok",
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        predictions=_prediction_frame(
            model_key="arimax",
            label=MODEL_CATALOG["arimax"]["label"],
            validation_true=val,
            validation_pred=val_pred,
            test_true=test,
            test_pred=test_pred,
        ),
        parameters={"order": config.arima_order},
    )


def _evaluate_prophet(dataset_parts: dict[str, Any]) -> ArenaModelResult:
    train = dataset_parts["train"]
    val = dataset_parts["val"]
    test = dataset_parts["test"]
    val_pred = _fit_predict_prophet(train, val.index)
    test_pred = _fit_predict_prophet(pd.concat([train, val]), test.index)
    validation_metrics = regression_report(val.values, val_pred.values)
    test_metrics = regression_report(test.values, test_pred.values)
    return ArenaModelResult(
        model_key="prophet",
        label=MODEL_CATALOG["prophet"]["label"],
        family=MODEL_CATALOG["prophet"]["family"],
        status="ok",
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        predictions=_prediction_frame(
            model_key="prophet",
            label=MODEL_CATALOG["prophet"]["label"],
            validation_true=val,
            validation_pred=val_pred,
            test_true=test,
            test_pred=test_pred,
        ),
    )


def _evaluate_tabular(dataset_parts: dict[str, Any], config: ArenaRunConfig, model_key: ModelKey) -> ArenaModelResult:
    train = dataset_parts["train"]
    val = dataset_parts["val"]
    test = dataset_parts["test"]
    val_pred, val_info = _fit_predict_tabular(
        model_key=model_key,
        train=train,
        evaluate=val,
        lags=config.lags,
        random_state=config.random_seed,
    )
    test_pred, test_info = _fit_predict_tabular(
        model_key=model_key,
        train=pd.concat([train, val]),
        evaluate=test,
        lags=config.lags,
        random_state=config.random_seed,
    )
    validation_metrics = regression_report(val.values, val_pred.values)
    test_metrics = regression_report(test.values, test_pred.values)
    return ArenaModelResult(
        model_key=model_key,
        label=MODEL_CATALOG[model_key]["label"],
        family=MODEL_CATALOG[model_key]["family"],
        status="ok",
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        predictions=_prediction_frame(
            model_key=model_key,
            label=MODEL_CATALOG[model_key]["label"],
            validation_true=val,
            validation_pred=val_pred,
            test_true=test,
            test_pred=test_pred,
        ),
        parameters={"lags": config.lags, "validation_setup": val_info, "test_setup": test_info},
    )


def _evaluate_sequence(dataset_parts: dict[str, Any], config: ArenaRunConfig, model_key: ModelKey) -> ArenaModelResult:
    train = dataset_parts["train"]
    val = dataset_parts["val"]
    test = dataset_parts["test"]

    if model_key == "lstm":
        builder = build_lstm_model
        builder_params = {
            "units": config.lstm_units,
            "dense_units": config.lstm_dense_units,
            "dropout": config.dropout,
            "learning_rate": config.learning_rate,
        }
    else:
        builder = build_gru_model
        builder_params = {
            "units": config.gru_units,
            "dense_units": config.gru_dense_units,
            "dropout": config.dropout,
            "learning_rate": config.learning_rate,
        }

    val_pred, val_history, val_info = _fit_predict_sequence(
        builder=builder,
        train=train,
        evaluate=val,
        look_back=config.look_back,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_split=config.validation_split,
        patience=config.patience,
        random_seed=config.random_seed,
        builder_params=builder_params,
    )
    test_pred, test_history, test_info = _fit_predict_sequence(
        builder=builder,
        train=pd.concat([train, val]),
        evaluate=test,
        look_back=config.look_back,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_split=config.validation_split,
        patience=config.patience,
        random_seed=config.random_seed,
        builder_params=builder_params,
    )
    validation_metrics = regression_report(val.values, val_pred.values)
    test_metrics = regression_report(test.values, test_pred.values)
    history = pd.concat(
        [
            val_history.assign(phase="validation_fit", epoch=lambda frame: frame.index + 1),
            test_history.assign(phase="test_fit", epoch=lambda frame: frame.index + 1),
        ],
        ignore_index=True,
    )
    return ArenaModelResult(
        model_key=model_key,
        label=MODEL_CATALOG[model_key]["label"],
        family=MODEL_CATALOG[model_key]["family"],
        status="ok",
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        predictions=_prediction_frame(
            model_key=model_key,
            label=MODEL_CATALOG[model_key]["label"],
            validation_true=val,
            validation_pred=val_pred,
            test_true=test,
            test_pred=test_pred,
        ),
        parameters={"look_back": config.look_back, "validation_setup": val_info, "test_setup": test_info},
        history=history,
    )


def run_model_arena(dataset: ArenaDataset, model_keys: list[ModelKey], config: ArenaRunConfig) -> list[ArenaModelResult]:
    dataset_parts = _split_dataset(dataset, config)
    availability = available_model_catalog(has_exog=bool(dataset.exog_cols))
    results: list[ArenaModelResult] = []

    for model_key in model_keys:
        label = MODEL_CATALOG[model_key]["label"]
        family = MODEL_CATALOG[model_key]["family"]
        start = perf_counter()
        if not availability[model_key]["available"]:
            results.append(
                ArenaModelResult(
                    model_key=model_key,
                    label=label,
                    family=family,
                    status="skipped",
                    error=availability[model_key]["reason"],
                )
            )
            continue

        try:
            if model_key in {"arima", "arma", "sarima"}:
                result = _evaluate_arima_like(dataset_parts, config, model_key)
            elif model_key == "arimax":
                result = _evaluate_arimax(dataset_parts, config)
            elif model_key == "prophet":
                result = _evaluate_prophet(dataset_parts)
            elif model_key in {"linear_regression", "random_forest", "gradient_boosting", "xgboost"}:
                result = _evaluate_tabular(dataset_parts, config, model_key)
            elif model_key in {"lstm", "gru"}:
                result = _evaluate_sequence(dataset_parts, config, model_key)
            else:
                raise ValueError(f"Unsupported model '{model_key}'.")
            result.duration_seconds = perf_counter() - start
            results.append(result)
        except Exception as exc:
            results.append(
                ArenaModelResult(
                    model_key=model_key,
                    label=label,
                    family=family,
                    status="error",
                    duration_seconds=perf_counter() - start,
                    error=str(exc),
                )
            )
    return results


def build_leaderboard_frame(results: list[ArenaModelResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        row = {
            "model_key": result.model_key,
            "model": result.label,
            "family": result.family,
            "status": result.status,
            "duration_seconds": result.duration_seconds,
            "error": result.error,
        }
        if result.validation_metrics:
            row.update({f"validation_{metric}": value for metric, value in result.validation_metrics.items()})
        if result.test_metrics:
            row.update({f"test_{metric}": value for metric, value in result.test_metrics.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def build_predictions_frame(results: list[ArenaModelResult]) -> pd.DataFrame:
    frames = [result.predictions for result in results if result.predictions is not None]
    if not frames:
        return pd.DataFrame(columns=["actual", "predicted", "phase", "model_key", "model"])
    return pd.concat(frames).sort_index()
