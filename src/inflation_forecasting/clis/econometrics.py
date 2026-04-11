from __future__ import annotations

import argparse

import pandas as pd

from ..models.econometria.arch_garch import fit_arch, fit_garch
from ..models.econometria.arima import evaluate_arima, fit_arima, forecast_arima, grid_search_arima
from ..models.econometria.arimax import evaluate_arimax
from ..models.econometria.arma import evaluate_arma, grid_search_arma
from ..models.econometria.prophet import train_prophet
from ..models.econometria.sarima import evaluate_sarima, infer_seasonal_period
from .common import (
    create_run,
    load_series,
    load_state_frame,
    parse_cols,
    parse_order,
    save_json_artifact,
    save_table_artifact,
    write_manifest,
)


def _leaderboard(results: list) -> pd.DataFrame:
    rows = []
    for result in results:
        row = {"order": str(getattr(result, "order", None))}
        if hasattr(result, "seasonal_order"):
            row["seasonal_order"] = str(result.seasonal_order)
        row.update(result.metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def cmd_arima(args: argparse.Namespace) -> None:
    series = load_series(args)
    run = create_run(args, "arima")
    artifact_paths: dict[str, object] = {}

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
            raise RuntimeError("No ARIMA models fit successfully.")
        best = results[0]
        artifact_paths["leaderboard"] = save_table_artifact(run, "leaderboard", _leaderboard(results))
        artifact_paths["predictions"] = save_table_artifact(run, "predictions", best.predictions)
        metrics = best.metrics
        model_name = "arima_grid_search"
        print(f"Best order: {best.order}")
        print(metrics)
    else:
        order = parse_order(args.order, 3)
        result = evaluate_arima(series, order=order, test_size=args.test_size)
        metrics = result.metrics
        artifact_paths["predictions"] = save_table_artifact(run, "predictions", result.predictions)
        print(metrics)
        if args.forecast_steps:
            model_fit = fit_arima(series, order=order)
            future = forecast_arima(model_fit, steps=args.forecast_steps)
            artifact_paths["forecast"] = save_table_artifact(run, "forecast", future)
        model_name = "arima"

    artifact_paths["metrics"] = save_json_artifact(run, "metrics", metrics)
    artifact_paths["manifest"] = write_manifest(
        args=args,
        run=run,
        command_name="arima",
        model_family="econometrics",
        model_name=model_name,
        series=series,
        metrics=metrics,
        artifact_paths=artifact_paths,
    )
    print(f"Saved run artifacts to {run.path}")


def cmd_arma(args: argparse.Namespace) -> None:
    series = load_series(args)
    run = create_run(args, "arma")
    artifact_paths: dict[str, object] = {}

    if args.grid:
        results = grid_search_arma(
            series,
            p_range=range(args.p_min, args.p_max + 1),
            q_range=range(args.q_min, args.q_max + 1),
            test_size=args.test_size,
            max_models=args.max_models,
        )
        if not results:
            raise RuntimeError("No ARMA models fit successfully.")
        best = results[0]
        artifact_paths["leaderboard"] = save_table_artifact(run, "leaderboard", _leaderboard(results))
        artifact_paths["predictions"] = save_table_artifact(run, "predictions", best.predictions)
        metrics = best.metrics
        model_name = "arma_grid_search"
        print(f"Best order: {best.order}")
        print(metrics)
    else:
        order = parse_order(args.order, 2)
        result = evaluate_arma(series, order=order, test_size=args.test_size)
        metrics = result.metrics
        artifact_paths["predictions"] = save_table_artifact(run, "predictions", result.predictions)
        model_name = "arma"
        print(metrics)

    artifact_paths["metrics"] = save_json_artifact(run, "metrics", metrics)
    artifact_paths["manifest"] = write_manifest(
        args=args,
        run=run,
        command_name="arma",
        model_family="econometrics",
        model_name=model_name,
        series=series,
        metrics=metrics,
        artifact_paths=artifact_paths,
    )
    print(f"Saved run artifacts to {run.path}")


def cmd_sarima(args: argparse.Namespace) -> None:
    series = load_series(args)
    order = parse_order(args.order, 3)

    if args.seasonal_order:
        seasonal_order = parse_order(args.seasonal_order, 4)
    else:
        seasonal_period = args.seasonal_period
        if args.auto_seasonal and not seasonal_period:
            seasonal_period = infer_seasonal_period(series)
        if seasonal_period:
            seasonal_order = (args.seasonal_p, args.seasonal_d, args.seasonal_q, seasonal_period)
        else:
            seasonal_order = (0, 0, 0, 0)

    result = evaluate_sarima(series, order=order, seasonal_order=seasonal_order, test_size=args.test_size)
    run = create_run(args, "sarima")
    artifact_paths = {
        "metrics": save_json_artifact(run, "metrics", result.metrics),
        "predictions": save_table_artifact(run, "predictions", result.predictions),
    }
    artifact_paths["manifest"] = write_manifest(
        args=args,
        run=run,
        command_name="sarima",
        model_family="econometrics",
        model_name="sarima",
        series=series,
        metrics=result.metrics,
        artifact_paths=artifact_paths,
        extra={"seasonal_order": seasonal_order},
    )
    print(result.metrics)
    print(f"Saved run artifacts to {run.path}")


def cmd_arimax(args: argparse.Namespace) -> None:
    exog_cols = parse_cols(args.exog_cols)
    if not exog_cols:
        raise ValueError("ARIMAX requires --exog-cols with comma-separated column names.")

    df_state = load_state_frame(args, exog_cols=exog_cols)
    missing = [column for column in exog_cols if column not in df_state.columns]
    if missing:
        raise ValueError(f"Exogenous columns not found in data: {missing}")

    series = df_state[args.target]
    exog = df_state[exog_cols]
    order = parse_order(args.order, 3)
    result = evaluate_arimax(series, exog, order=order, test_size=args.test_size)

    run = create_run(args, "arimax")
    artifact_paths = {
        "metrics": save_json_artifact(run, "metrics", result.metrics),
        "predictions": save_table_artifact(run, "predictions", result.predictions),
    }
    artifact_paths["manifest"] = write_manifest(
        args=args,
        run=run,
        command_name="arimax",
        model_family="econometrics",
        model_name="arimax",
        series=series,
        metrics=result.metrics,
        artifact_paths=artifact_paths,
        extra={"exogenous_columns": exog_cols},
    )
    print(result.metrics)
    print(f"Saved run artifacts to {run.path}")


def cmd_arch(args: argparse.Namespace) -> None:
    series = load_series(args)
    result = fit_arch(series, p=args.p, mean=args.mean)
    metrics = {"aic": result.aic, "bic": result.bic}
    run = create_run(args, "arch")
    artifact_paths = {
        "metrics": save_json_artifact(run, "metrics", metrics),
        "conditional_volatility": save_table_artifact(run, "conditional_volatility", result.conditional_volatility),
    }
    artifact_paths["manifest"] = write_manifest(
        args=args,
        run=run,
        command_name="arch",
        model_family="econometrics",
        model_name="arch",
        series=series,
        metrics=metrics,
        artifact_paths=artifact_paths,
    )
    print(metrics)
    print(f"Saved run artifacts to {run.path}")


def cmd_garch(args: argparse.Namespace) -> None:
    series = load_series(args)
    result = fit_garch(series, p=args.p, q=args.q, mean=args.mean)
    metrics = {"aic": result.aic, "bic": result.bic}
    run = create_run(args, "garch")
    artifact_paths = {
        "metrics": save_json_artifact(run, "metrics", metrics),
        "conditional_volatility": save_table_artifact(run, "conditional_volatility", result.conditional_volatility),
    }
    artifact_paths["manifest"] = write_manifest(
        args=args,
        run=run,
        command_name="garch",
        model_family="econometrics",
        model_name="garch",
        series=series,
        metrics=metrics,
        artifact_paths=artifact_paths,
    )
    print(metrics)
    print(f"Saved run artifacts to {run.path}")


def cmd_prophet(args: argparse.Namespace) -> None:
    series = load_series(args)
    result = train_prophet(series, test_size=args.test_size)
    run = create_run(args, "prophet")
    artifact_paths = {
        "metrics": save_json_artifact(run, "metrics", result.metrics),
        "predictions": save_table_artifact(run, "predictions", result.predictions),
    }
    if args.forecast_steps:
        future = result.model.predict(result.model.make_future_dataframe(periods=args.forecast_steps, freq="Q")).tail(
            args.forecast_steps
        )
        artifact_paths["forecast"] = save_table_artifact(run, "forecast", future.set_index("ds")["yhat"])

    artifact_paths["manifest"] = write_manifest(
        args=args,
        run=run,
        command_name="prophet",
        model_family="econometrics",
        model_name="prophet",
        series=series,
        metrics=result.metrics,
        artifact_paths=artifact_paths,
    )
    print(result.metrics)
    print(f"Saved run artifacts to {run.path}")


def register_econometrics_commands(subparsers, common: argparse.ArgumentParser) -> None:
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

    arma = subparsers.add_parser("arma", parents=[common], help="ARMA training")
    arma.add_argument("--order", default="1,1")
    arma.add_argument("--test-size", type=float, default=0.2)
    arma.add_argument("--grid", action="store_true")
    arma.add_argument("--p-min", type=int, default=0)
    arma.add_argument("--p-max", type=int, default=3)
    arma.add_argument("--q-min", type=int, default=0)
    arma.add_argument("--q-max", type=int, default=3)
    arma.add_argument("--max-models", type=int, default=None)
    arma.set_defaults(func=cmd_arma)

    sarima = subparsers.add_parser("sarima", parents=[common], help="SARIMA training")
    sarima.add_argument("--order", default="1,0,1")
    sarima.add_argument("--test-size", type=float, default=0.2)
    sarima.add_argument("--seasonal-order", default=None)
    sarima.add_argument("--seasonal-p", type=int, default=1)
    sarima.add_argument("--seasonal-d", type=int, default=0)
    sarima.add_argument("--seasonal-q", type=int, default=1)
    sarima.add_argument("--seasonal-period", type=int, default=0)
    sarima.add_argument("--auto-seasonal", action="store_true")
    sarima.set_defaults(func=cmd_sarima)

    arimax = subparsers.add_parser("arimax", parents=[common], help="ARIMAX training")
    arimax.add_argument("--order", default="1,0,1")
    arimax.add_argument("--test-size", type=float, default=0.2)
    arimax.add_argument("--exog-cols", default=None)
    arimax.set_defaults(func=cmd_arimax)

    arch = subparsers.add_parser("arch", parents=[common], help="ARCH volatility model")
    arch.add_argument("--p", type=int, default=1)
    arch.add_argument("--mean", default="constant")
    arch.set_defaults(func=cmd_arch)

    garch = subparsers.add_parser("garch", parents=[common], help="GARCH volatility model")
    garch.add_argument("--p", type=int, default=1)
    garch.add_argument("--q", type=int, default=1)
    garch.add_argument("--mean", default="constant")
    garch.set_defaults(func=cmd_garch)

    prophet = subparsers.add_parser("prophet", parents=[common], help="Train Prophet")
    prophet.add_argument("--test-size", type=float, default=0.2)
    prophet.add_argument("--forecast-steps", type=int, default=0)
    prophet.set_defaults(func=cmd_prophet)
