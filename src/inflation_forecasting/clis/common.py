from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from ..artifacts.manifests import build_run_manifest
from ..artifacts.storage import RunDirectory, create_run_directory, relative_artifact_path
from ..data.io import DEFAULT_RAW_PATH, load_raw_data, save_dataframe, save_json, save_yaml
from ..data.preprocessing import add_quarterly_date, filter_state, prepare_state_series


def parse_order(text: str, expected: int) -> tuple[int, ...]:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != expected:
        raise ValueError(f"Expected {expected} values, got {len(parts)} in '{text}'.")
    return tuple(int(part) for part in parts)


def parse_cols(text: str | None) -> list[str]:
    if not text:
        return []
    return [column.strip() for column in text.split(",") if column.strip()]


def command_parameters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: value
        for key, value in vars(args).items()
        if key not in {"func"}
    }


def create_run(args: argparse.Namespace, command_name: str) -> RunDirectory:
    return create_run_directory(command_name, output_dir=args.output_dir)


def load_state_frame(args: argparse.Namespace, exog_cols: list[str] | None = None) -> pd.DataFrame:
    df = load_raw_data(args.data_path)
    df = add_quarterly_date(
        df,
        year_col=args.year_col,
        quarter_col=args.quarter_col,
        date_col=args.date_col,
    )
    df_state = filter_state(df, state=args.state)
    df_state = df_state.set_index(args.date_col).sort_index()
    if not args.no_interpolate:
        columns = [args.target] + (exog_cols or [])
        for column in columns:
            if column in df_state.columns:
                df_state[column] = df_state[column].interpolate()
    return df_state


def load_series(args: argparse.Namespace) -> pd.Series:
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


def save_table_artifact(run: RunDirectory, artifact_name: str, data: pd.DataFrame | pd.Series) -> Path:
    frame = data.to_frame(name=data.name or artifact_name) if isinstance(data, pd.Series) else data
    return save_dataframe(frame, run.path / f"{artifact_name}.csv")


def save_json_artifact(run: RunDirectory, artifact_name: str, payload: dict[str, Any]) -> Path:
    return save_json(payload, run.path / f"{artifact_name}.json")


def save_yaml_artifact(run: RunDirectory, artifact_name: str, payload: dict[str, Any]) -> Path:
    return save_yaml(payload, run.path / f"{artifact_name}.yml")


def write_manifest(
    *,
    args: argparse.Namespace,
    run: RunDirectory,
    command_name: str,
    model_family: str,
    model_name: str,
    series: pd.Series | None,
    metrics: dict[str, Any] | None,
    artifact_paths: dict[str, Path],
    extra: dict[str, Any] | None = None,
) -> Path:
    manifest = build_run_manifest(
        run_id=run.run_id,
        command_name=command_name,
        model_family=model_family,
        model_name=model_name,
        parameters=command_parameters(args),
        metrics=metrics,
        artifacts={name: relative_artifact_path(path, run.path) for name, path in artifact_paths.items()},
        series=series,
        data_source=args.data_path or DEFAULT_RAW_PATH,
        state=getattr(args, "state", None),
        target=getattr(args, "target", None),
        extra=extra,
    )
    return save_yaml(manifest, run.path / "manifest.yml")


def add_common_args(common: argparse.ArgumentParser) -> None:
    common.add_argument("--data-path", default=None)
    common.add_argument("--state", default="Maryland")
    common.add_argument("--target", default="pi")
    common.add_argument("--year-col", default="year")
    common.add_argument("--quarter-col", default="quarter")
    common.add_argument("--date-col", default="date")
    common.add_argument("--no-interpolate", action="store_true")
    common.add_argument("--output-dir", default=None)
