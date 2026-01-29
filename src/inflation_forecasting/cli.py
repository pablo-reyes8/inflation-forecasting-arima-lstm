from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from . import __version__
from .decomposition import hp_filter, seasonal_decompose_series
from .io import default_outputs_dir, load_raw_data, save_dataframe, save_json
from .metrics import regression_report
from .preprocess import add_quarterly_date, filter_state, prepare_state_series, summary_stats
from .models.econometria.arima import evaluate_arima, grid_search_arima
from .models.econometria.arma import evaluate_arma, grid_search_arma
from .models.econometria.sarima import evaluate_sarima, infer_seasonal_period
from .models.econometria.arimax import evaluate_arimax
from .models.econometria.arch_garch import fit_arch, fit_garch
from .models.ml.lstm import train_lstm, tune_lstm, forecast_future, save_lstm_artifacts, load_lstm_artifacts
from .models.ml.gru import train_gru
from .models.econometria.prophet import train_prophet
from .models.ml.baselines import train_ml_model


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_order(text: str, expected: int) -> tuple[int, ...]:
    parts = [p.strip() for p in text.split(",") if p.strip()]\n+    if len(parts) != expected:\n+        raise ValueError(f\"Expected {expected} values, got {len(parts)} in '{text}'.\")\n+    return tuple(int(p) for p in parts)\n+\n+\n+def _parse_cols(text: str | None) -> list[str]:\n+    if not text:\n+        return []\n+    return [c.strip() for c in text.split(\",\") if c.strip()]\n+\n+\n+def _load_state_frame(args, exog_cols: list[str] | None = None):\n+    df = load_raw_data(args.data_path)\n+    df = add_quarterly_date(df, year_col=args.year_col, quarter_col=args.quarter_col, date_col=args.date_col)\n+    df_state = filter_state(df, state=args.state)\n+    df_state = df_state.set_index(args.date_col).sort_index()\n+    if not args.no_interpolate:\n+        cols = [args.target] + (exog_cols or [])\n+        for col in cols:\n+            if col in df_state.columns:\n+                df_state[col] = df_state[col].interpolate()\n+    return df_state\n+

def _load_series(args):
    df = load_raw_data(args.data_path)
    return prepare_state_series(
        df,
        state=args.state,
        target=args.target,
        year_col=args.year_col,
        quarter_col=args.quarter_col,
        date_col=args.date_col,
        interpolate=not args.no_interpolate,
    )


def cmd_describe(args):
    series = _load_series(args)
    stats = summary_stats(series)
    out_dir = Path(args.output_dir) if args.output_dir else default_outputs_dir()
    path = save_dataframe(stats, out_dir / f"summary_{args.state}_{_timestamp()}.csv")
    print(stats.to_string())
    print(f"Saved summary to {path}")


def cmd_hp_filter(args):
    series = _load_series(args)
    trend, cycle = hp_filter(series, lamb=args.lamb)
    out = trend.to_frame("trend")
    out["cycle"] = cycle
    out_dir = Path(args.output_dir) if args.output_dir else default_outputs_dir()
    path = save_dataframe(out, out_dir / f"hp_filter_{args.state}_{_timestamp()}.csv")
    print(f"Saved HP filter output to {path}")


def cmd_decompose(args):
    series = _load_series(args)
    result = seasonal_decompose_series(series, model=args.model, period=args.period)
    out = result.trend.to_frame("trend")
    out["seasonal"] = result.seasonal
    out["resid"] = result.resid
    out_dir = Path(args.output_dir) if args.output_dir else default_outputs_dir()
    path = save_dataframe(out, out_dir / f"decompose_{args.state}_{_timestamp()}.csv")
    print(f"Saved decomposition output to {path}")


def cmd_arima(args):
    series = _load_series(args)
    out_dir = Path(args.output_dir) if args.output_dir else default_outputs_dir()
    if args.grid:
        results = grid_search_arima(
            series,
            p_range=range(args.p_min, args.p_max + 1),
            d_range=range(args.d_min, args.d_max + 1),
            q_range=range(args.q_min, args.q_max + 1),
            test_size=args.test_size,
            max_models=args.max_models,
        )
        if not results:
            print("No ARIMA models fit successfully.")
            return
        best = results[0]
        metrics_path = save_json(best.metrics, out_dir / f"arima_best_metrics_{_timestamp()}.json")
        preds_path = save_dataframe(best.predictions.to_frame("prediction"), out_dir / f"arima_best_preds_{_timestamp()}.csv")
        print(f"Best order: {best.order}")
        print(best.metrics)
        print(f"Saved metrics to {metrics_path}")
        print(f"Saved predictions to {preds_path}")
    else:
        order = tuple(int(x) for x in args.order.split(","))
        res = evaluate_arima(series, order=order, test_size=args.test_size)
        metrics_path = save_json(res.metrics, out_dir / f"arima_metrics_{_timestamp()}.json")
        preds_path = save_dataframe(res.predictions.to_frame("prediction"), out_dir / f"arima_preds_{_timestamp()}.csv")
        print(res.metrics)
        print(f"Saved metrics to {metrics_path}")
        print(f"Saved predictions to {preds_path}")
        if args.forecast_steps:
            from .models.econometria.arima import fit_arima, forecast_arima

            model_fit = fit_arima(series, order=order)
            future = forecast_arima(model_fit, steps=args.forecast_steps)
            future_path = save_dataframe(future.to_frame("forecast"), out_dir / f"arima_forecast_{_timestamp()}.csv")
            print(f"Saved forecast to {future_path}")


def cmd_lstm(args):
    series = _load_series(args)
    res = train_lstm(
        series,
        look_back=args.look_back,
        test_size=args.test_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        units=args.units,
        dense_units=args.dense_units,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        verbose=args.verbose,
    )
    out_dir = Path(args.output_dir) if args.output_dir else default_outputs_dir()
    metrics_path = save_json(res.metrics, out_dir / f"lstm_metrics_{_timestamp()}.json")
    preds_path = save_dataframe(res.predictions, out_dir / f"lstm_preds_{_timestamp()}.csv")
    print(res.metrics)
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {preds_path}")

    if args.forecast_steps:
        future = forecast_future(res.model, res.scaler, series, look_back=args.look_back, steps=args.forecast_steps)
        future_path = save_dataframe(future.to_frame("forecast"), out_dir / f"lstm_forecast_{_timestamp()}.csv")
        print(f"Saved forecast to {future_path}")

    if args.save_model:
        model_path = out_dir / f"lstm_model_{_timestamp()}.keras"
        scaler_path = out_dir / f"lstm_scaler_{_timestamp()}.joblib"
        save_lstm_artifacts(res.model, res.scaler, str(model_path), str(scaler_path))
        print(f"Saved model to {model_path}")
        print(f"Saved scaler to {scaler_path}")


def cmd_lstm_tune(args):
    series = _load_series(args)
    result = tune_lstm(
        series,
        look_back=args.look_back,
        test_size=args.test_size,
        max_trials=args.max_trials,
        executions_per_trial=args.executions_per_trial,
        directory=args.directory,
        project_name=args.project_name,
    )
    out_dir = Path(args.output_dir) if args.output_dir else default_outputs_dir()
    metrics_path = save_json(result.best_hyperparameters, out_dir / f"lstm_best_hp_{_timestamp()}.json")
    print(result.best_hyperparameters)
    print(f"Saved best hyperparameters to {metrics_path}")


def cmd_gru(args):
    series = _load_series(args)
    res = train_gru(
        series,
        look_back=args.look_back,
        test_size=args.test_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        units=args.units,
        dense_units=args.dense_units,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        verbose=args.verbose,
    )
    out_dir = Path(args.output_dir) if args.output_dir else default_outputs_dir()
    metrics_path = save_json(res.metrics, out_dir / f"gru_metrics_{_timestamp()}.json")
    preds_path = save_dataframe(res.predictions, out_dir / f"gru_preds_{_timestamp()}.csv")
    print(res.metrics)
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {preds_path}")


def cmd_prophet(args):
    series = _load_series(args)
    res = train_prophet(series, test_size=args.test_size)
    out_dir = Path(args.output_dir) if args.output_dir else default_outputs_dir()
    metrics_path = save_json(res.metrics, out_dir / f"prophet_metrics_{_timestamp()}.json")
    preds_path = save_dataframe(res.predictions, out_dir / f"prophet_preds_{_timestamp()}.csv")
    print(res.metrics)
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {preds_path}")
    if args.forecast_steps:
        from prophet import Prophet

        df = series.reset_index()
        df.columns = ["ds", "y"]
        model = Prophet()
        model.fit(df)
        future_df = model.make_future_dataframe(periods=args.forecast_steps, freq="Q")
        forecast = model.predict(future_df).tail(args.forecast_steps)
        future = forecast.set_index("ds")["yhat"]
        future_path = save_dataframe(future.to_frame("forecast"), out_dir / f"prophet_forecast_{_timestamp()}.csv")
        print(f"Saved forecast to {future_path}")


def cmd_lstm_forecast(args):
    series = _load_series(args)
    model, scaler = load_lstm_artifacts(args.model_path, args.scaler_path)
    future = forecast_future(model, scaler, series, look_back=args.look_back, steps=args.steps)
    out_dir = Path(args.output_dir) if args.output_dir else default_outputs_dir()
    future_path = save_dataframe(future.to_frame("forecast"), out_dir / f"lstm_forecast_{_timestamp()}.csv")
    print(f"Saved forecast to {future_path}")


def cmd_ml(args):
    series = _load_series(args)
    res = train_ml_model(
        series,
        model_name=args.model,
        lags=args.lags,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    out_dir = Path(args.output_dir) if args.output_dir else default_outputs_dir()
    metrics_path = save_json(res.metrics, out_dir / f"{args.model}_metrics_{_timestamp()}.json")
    preds_path = save_dataframe(res.predictions, out_dir / f"{args.model}_preds_{_timestamp()}.csv")
    print(res.metrics)
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {preds_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inflation-forecast", description="Inflation forecasting toolkit")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-path", default=None)
    common.add_argument("--state", default="Maryland")
    common.add_argument("--target", default="pi")
    common.add_argument("--year-col", default="year")
    common.add_argument("--quarter-col", default="quarter")
    common.add_argument("--date-col", default="date")
    common.add_argument("--no-interpolate", action="store_true")
    common.add_argument("--output-dir", default=None)

    subparsers = parser.add_subparsers(dest="command", required=True)

    describe = subparsers.add_parser("describe", parents=[common], help="Descriptive stats")
    describe.set_defaults(func=cmd_describe)

    hp = subparsers.add_parser("hp-filter", parents=[common], help="HP filter trend/cycle")
    hp.add_argument("--lamb", type=float, default=1600.0)
    hp.set_defaults(func=cmd_hp_filter)

    decomp = subparsers.add_parser("decompose", parents=[common], help="Seasonal decomposition")
    decomp.add_argument("--model", default="additive")
    decomp.add_argument("--period", type=int, default=4)
    decomp.set_defaults(func=cmd_decompose)

    arima = subparsers.add_parser("arima", parents=[common], help="ARIMA training")
    arima.add_argument("--order", default="1,0,3")
    arima.add_argument("--test-size", type=float, default=0.2)
    arima.add_argument("--forecast-steps", type=int, default=0)
    arima.add_argument("--grid", action="store_true")
    arima.add_argument("--p-min", type=int, default=0)
    arima.add_argument("--p-max", type=int, default=3)
    arima.add_argument("--d-min", type=int, default=0)
    arima.add_argument("--d-max", type=int, default=1)
    arima.add_argument("--q-min", type=int, default=0)
    arima.add_argument("--q-max", type=int, default=3)
    arima.add_argument("--max-models", type=int, default=None)
    arima.set_defaults(func=cmd_arima)

    lstm = subparsers.add_parser("lstm-train", parents=[common], help="Train LSTM")
    lstm.add_argument("--look-back", type=int, default=4)
    lstm.add_argument("--test-size", type=float, default=0.2)
    lstm.add_argument("--epochs", type=int, default=80)
    lstm.add_argument("--batch-size", type=int, default=16)
    lstm.add_argument("--units", type=int, default=100)
    lstm.add_argument("--dense-units", type=int, default=64)
    lstm.add_argument("--dropout", type=float, default=0.2)
    lstm.add_argument("--learning-rate", type=float, default=1e-3)
    lstm.add_argument("--forecast-steps", type=int, default=0)
    lstm.add_argument("--save-model", action="store_true")
    lstm.add_argument("--verbose", type=int, default=0)
    lstm.set_defaults(func=cmd_lstm)

    lstm_tune = subparsers.add_parser("lstm-tune", parents=[common], help="Tune LSTM hyperparameters")
    lstm_tune.add_argument("--look-back", type=int, default=4)
    lstm_tune.add_argument("--test-size", type=float, default=0.2)
    lstm_tune.add_argument("--max-trials", type=int, default=10)
    lstm_tune.add_argument("--executions-per-trial", type=int, default=2)
    lstm_tune.add_argument("--directory", default="tuning")
    lstm_tune.add_argument("--project-name", default="lstm")
    lstm_tune.set_defaults(func=cmd_lstm_tune)

    gru = subparsers.add_parser("gru-train", parents=[common], help="Train GRU")
    gru.add_argument("--look-back", type=int, default=4)
    gru.add_argument("--test-size", type=float, default=0.2)
    gru.add_argument("--epochs", type=int, default=80)
    gru.add_argument("--batch-size", type=int, default=16)
    gru.add_argument("--units", type=int, default=64)
    gru.add_argument("--dense-units", type=int, default=32)
    gru.add_argument("--dropout", type=float, default=0.2)
    gru.add_argument("--learning-rate", type=float, default=1e-3)
    gru.add_argument("--verbose", type=int, default=0)
    gru.set_defaults(func=cmd_gru)

    prophet = subparsers.add_parser("prophet", parents=[common], help="Train Prophet")
    prophet.add_argument("--test-size", type=float, default=0.2)
    prophet.add_argument("--forecast-steps", type=int, default=0)
    prophet.set_defaults(func=cmd_prophet)

    lstm_forecast = subparsers.add_parser("lstm-forecast", parents=[common], help="Forecast with saved LSTM")
    lstm_forecast.add_argument("--model-path", required=True)
    lstm_forecast.add_argument("--scaler-path", required=True)
    lstm_forecast.add_argument("--look-back", type=int, default=4)
    lstm_forecast.add_argument("--steps", type=int, default=4)
    lstm_forecast.set_defaults(func=cmd_lstm_forecast)

    ml = subparsers.add_parser("ml-train", parents=[common], help="Train ML baseline")
    ml.add_argument("--model", default="random_forest", choices=["random_forest", "gradient_boosting", "xgboost", "linear_regression"])
    ml.add_argument("--lags", type=int, default=4)
    ml.add_argument("--test-size", type=float, default=0.2)
    ml.add_argument("--random-state", type=int, default=42)
    ml.set_defaults(func=cmd_ml)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
