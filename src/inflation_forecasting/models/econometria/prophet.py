from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ...metrics import regression_report


def _require_prophet():
    try:
        from prophet import Prophet  # noqa: F401
    except ImportError as exc:
        raise ImportError("prophet is required for Prophet model. Install with `pip install prophet`.") from exc


@dataclass
class ProphetResult:
    model: object
    metrics: dict
    predictions: pd.DataFrame


def train_prophet(series: pd.Series, test_size: float = 0.2) -> ProphetResult:
    _require_prophet()
    from prophet import Prophet

    df = series.reset_index()
    df.columns = ["ds", "y"]

    n = len(df)
    split_idx = int(n * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    model = Prophet()
    model.fit(train_df)

    forecast = model.predict(test_df[["ds"]])
    preds = forecast["yhat"].values

    metrics = regression_report(test_df["y"].values, preds)

    pred_df = pd.DataFrame(
        {
            "y_true": test_df["y"].values,
            "y_pred": preds,
            "split": ["test"] * len(test_df),
        },
        index=test_df["ds"],
    )

    return ProphetResult(model=model, metrics=metrics, predictions=pred_df)
