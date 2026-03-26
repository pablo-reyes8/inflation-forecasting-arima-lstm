from __future__ import annotations

import argparse

from ..models.ml.gru import train_gru
from ..models.ml.lstm import (
    forecast_future,
    load_lstm_artifacts,
    save_lstm_artifacts,
    train_lstm,
    tune_lstm,
)
from .common import create_run, load_series, save_json_artifact, save_table_artifact, write_manifest


def cmd_lstm(args: argparse.Namespace) -> None:
    series = load_series(args)
    result = train_lstm(
        series,
        look_back=args.look_back,
        test_size=args.test_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        units=args.units,
        dense_units=args.dense_units,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        validation_split=args.validation_split,
        patience=args.patience,
        random_seed=args.random_seed,
        verbose=args.verbose,
    )

    run = create_run(args, "lstm-train")
    artifact_paths = {
        "metrics": save_json_artifact(run, "metrics", result.metrics),
        "predictions": save_table_artifact(run, "predictions", result.predictions),
        "history": save_table_artifact(run, "history", result.history_frame),
    }

    if args.forecast_steps:
        future = forecast_future(result.model, result.scaler, series, look_back=args.look_back, steps=args.forecast_steps)
        artifact_paths["forecast"] = save_table_artifact(run, "forecast", future)

    if args.save_model:
        model_path = run.path / "model.keras"
        scaler_path = run.path / "scaler.joblib"
        save_lstm_artifacts(result.model, result.scaler, str(model_path), str(scaler_path))
        artifact_paths["model"] = model_path
        artifact_paths["scaler"] = scaler_path

    artifact_paths["manifest"] = write_manifest(
        args=args,
        run=run,
        command_name="lstm-train",
        model_family="deep_learning",
        model_name="lstm",
        series=series,
        metrics=result.metrics,
        artifact_paths=artifact_paths,
    )
    print(result.metrics)
    print(f"Saved run artifacts to {run.path}")


def cmd_lstm_tune(args: argparse.Namespace) -> None:
    series = load_series(args)
    result = tune_lstm(
        series,
        look_back=args.look_back,
        test_size=args.test_size,
        max_trials=args.max_trials,
        executions_per_trial=args.executions_per_trial,
        directory=args.directory,
        project_name=args.project_name,
        epochs=args.epochs,
        random_seed=args.random_seed,
    )
    run = create_run(args, "lstm-tune")
    artifact_paths = {"best_hyperparameters": save_json_artifact(run, "best_hyperparameters", result.best_hyperparameters)}
    artifact_paths["manifest"] = write_manifest(
        args=args,
        run=run,
        command_name="lstm-tune",
        model_family="deep_learning",
        model_name="lstm_tuning",
        series=series,
        metrics={"trial_count": args.max_trials},
        artifact_paths=artifact_paths,
        extra={"best_hyperparameters": result.best_hyperparameters},
    )
    print(result.best_hyperparameters)
    print(f"Saved run artifacts to {run.path}")


def cmd_gru(args: argparse.Namespace) -> None:
    series = load_series(args)
    result = train_gru(
        series,
        look_back=args.look_back,
        test_size=args.test_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        units=args.units,
        dense_units=args.dense_units,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        validation_split=args.validation_split,
        patience=args.patience,
        random_seed=args.random_seed,
        verbose=args.verbose,
    )
    run = create_run(args, "gru-train")
    artifact_paths = {
        "metrics": save_json_artifact(run, "metrics", result.metrics),
        "predictions": save_table_artifact(run, "predictions", result.predictions),
        "history": save_table_artifact(run, "history", result.history_frame),
    }
    artifact_paths["manifest"] = write_manifest(
        args=args,
        run=run,
        command_name="gru-train",
        model_family="deep_learning",
        model_name="gru",
        series=series,
        metrics=result.metrics,
        artifact_paths=artifact_paths,
    )
    print(result.metrics)
    print(f"Saved run artifacts to {run.path}")


def cmd_lstm_forecast(args: argparse.Namespace) -> None:
    series = load_series(args)
    model, scaler = load_lstm_artifacts(args.model_path, args.scaler_path)
    future = forecast_future(model, scaler, series, look_back=args.look_back, steps=args.steps)
    run = create_run(args, "lstm-forecast")
    artifact_paths = {"forecast": save_table_artifact(run, "forecast", future)}
    artifact_paths["manifest"] = write_manifest(
        args=args,
        run=run,
        command_name="lstm-forecast",
        model_family="deep_learning",
        model_name="lstm_inference",
        series=series,
        metrics={"steps": args.steps},
        artifact_paths=artifact_paths,
        extra={"source_model_path": args.model_path, "source_scaler_path": args.scaler_path},
    )
    print(f"Saved run artifacts to {run.path}")


def register_neural_commands(subparsers, common: argparse.ArgumentParser) -> None:
    lstm = subparsers.add_parser("lstm-train", parents=[common], help="Train LSTM")
    lstm.add_argument("--look-back", type=int, default=4)
    lstm.add_argument("--test-size", type=float, default=0.2)
    lstm.add_argument("--epochs", type=int, default=80)
    lstm.add_argument("--batch-size", type=int, default=16)
    lstm.add_argument("--units", type=int, default=100)
    lstm.add_argument("--dense-units", type=int, default=64)
    lstm.add_argument("--dropout", type=float, default=0.2)
    lstm.add_argument("--learning-rate", type=float, default=1e-3)
    lstm.add_argument("--validation-split", type=float, default=0.1)
    lstm.add_argument("--patience", type=int, default=10)
    lstm.add_argument("--random-seed", type=int, default=42)
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
    lstm_tune.add_argument("--epochs", type=int, default=50)
    lstm_tune.add_argument("--random-seed", type=int, default=42)
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
    gru.add_argument("--validation-split", type=float, default=0.1)
    gru.add_argument("--patience", type=int, default=10)
    gru.add_argument("--random-seed", type=int, default=42)
    gru.add_argument("--verbose", type=int, default=0)
    gru.set_defaults(func=cmd_gru)

    lstm_forecast = subparsers.add_parser("lstm-forecast", parents=[common], help="Forecast with saved LSTM")
    lstm_forecast.add_argument("--model-path", required=True)
    lstm_forecast.add_argument("--scaler-path", required=True)
    lstm_forecast.add_argument("--look-back", type=int, default=4)
    lstm_forecast.add_argument("--steps", type=int, default=4)
    lstm_forecast.set_defaults(func=cmd_lstm_forecast)
