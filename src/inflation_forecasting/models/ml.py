from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from ..features import make_lag_features
from ..metrics import regression_report


ModelName = Literal["random_forest", "gradient_boosting", "xgboost", "linear_regression"]


def _require_sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise ImportError("scikit-learn is required for ML baselines. Install with `pip install scikit-learn`.") from exc


def _require_xgboost():
    try:
        import xgboost  # noqa: F401
    except ImportError as exc:
        raise ImportError("xgboost is required for XGBoost. Install with `pip install xgboost`.") from exc


@dataclass
class MLResult:
    model: object
    metrics: dict
    predictions: pd.DataFrame


def train_ml_model(
    series: pd.Series,
    model_name: ModelName = "random_forest",
    lags: int = 4,
    test_size: float = 0.2,
    random_state: int = 42,
) -> MLResult:
    _require_sklearn()
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression

    X, y = make_lag_features(series, lags=lags)
    n = len(X)
    split_idx = int(n * (1 - test_size))

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if model_name == "random_forest":
        model = RandomForestRegressor(n_estimators=300, random_state=random_state)
    elif model_name == "gradient_boosting":
        model = GradientBoostingRegressor(random_state=random_state)
    elif model_name == "linear_regression":
        model = LinearRegression()
    elif model_name == "xgboost":
        _require_xgboost()
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = regression_report(y_test.values, preds)

    pred_df = pd.DataFrame(
        {
            "y_true": y_test.values,
            "y_pred": preds,
            "split": ["test"] * len(y_test),
        },
        index=y_test.index,
    )

    return MLResult(model=model, metrics=metrics, predictions=pred_df)
