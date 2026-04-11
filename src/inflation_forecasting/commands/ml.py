from __future__ import annotations

import argparse

from ..models.ml.baselines import train_ml_model
from .common import create_run, load_series, save_json_artifact, save_table_artifact, write_manifest


def cmd_ml(args: argparse.Namespace) -> None:
    series = load_series(args)
    result = train_ml_model(
        series,
        model_name=args.model,
        lags=args.lags,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    run = create_run(args, "ml-train")
    artifact_paths = {
        "metrics": save_json_artifact(run, "metrics", result.metrics),
        "predictions": save_table_artifact(run, "predictions", result.predictions),
    }
    artifact_paths["manifest"] = write_manifest(
        args=args,
        run=run,
        command_name="ml-train",
        model_family="machine_learning",
        model_name=args.model,
        series=series,
        metrics=result.metrics,
        artifact_paths=artifact_paths,
    )
    print(result.metrics)
    print(f"Saved run artifacts to {run.path}")


def register_ml_commands(subparsers, common: argparse.ArgumentParser) -> None:
    ml = subparsers.add_parser("ml-train", parents=[common], help="Train ML baseline")
    ml.add_argument(
        "--model",
        default="random_forest",
        choices=["random_forest", "gradient_boosting", "xgboost", "linear_regression"],
    )
    ml.add_argument("--lags", type=int, default=4)
    ml.add_argument("--test-size", type=float, default=0.2)
    ml.add_argument("--random-state", type=int, default=42)
    ml.set_defaults(func=cmd_ml)
