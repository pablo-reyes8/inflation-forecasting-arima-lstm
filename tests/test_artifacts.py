from pathlib import Path

import pandas as pd

from inflation_forecasting.cli import build_parser
from inflation_forecasting.artifacts.manifests import build_run_manifest
from inflation_forecasting.artifacts.storage import relative_artifact_path
from inflation_forecasting.data.features import make_lag_features as data_make_lag_features
from inflation_forecasting.data.io import load_raw_data as data_load_raw_data
from inflation_forecasting.data.preprocessing import add_quarterly_date as data_add_quarterly_date
from inflation_forecasting.data.splits import train_test_split_series as data_train_test_split_series
from inflation_forecasting.datasets.features import make_lag_features as datasets_make_lag_features
from inflation_forecasting.datasets.io import load_raw_data as datasets_load_raw_data
from inflation_forecasting.datasets.preprocessing import add_quarterly_date as datasets_add_quarterly_date
from inflation_forecasting.datasets.splits import train_test_split_series as datasets_train_test_split_series
from inflation_forecasting.features import make_lag_features as flat_make_lag_features
from inflation_forecasting.io import load_raw_data as flat_load_raw_data
from inflation_forecasting.preprocess import add_quarterly_date as flat_add_quarterly_date
from inflation_forecasting.split import train_test_split_series as flat_train_test_split_series


def test_build_run_manifest_includes_series_metadata():
    series = pd.Series(
        [1.0, 1.2, 1.4, 1.5],
        index=pd.date_range("2020-03-31", periods=4, freq="QE-DEC"),
        name="pi",
    )
    manifest = build_run_manifest(
        run_id="run-001",
        command_name="arima",
        model_family="econometrics",
        model_name="arima",
        parameters={"order": "1,0,1"},
        metrics={"rmse": 0.2},
        artifacts={"metrics": "metrics.json"},
        series=series,
        data_source="Data/RawData.csv",
        state="Maryland",
        target="pi",
    )
    assert manifest["run"]["id"] == "run-001"
    assert manifest["data"]["n_observations"] == 4
    assert manifest["model"]["name"] == "arima"


def test_relative_artifact_path_prefers_run_relative_paths():
    run_dir = Path("/tmp/runs/run-001")
    artifact = run_dir / "metrics.json"
    assert relative_artifact_path(artifact, run_dir) == "metrics.json"


def test_compatibility_layers_point_to_dataset_namespace():
    assert flat_add_quarterly_date is datasets_add_quarterly_date
    assert data_add_quarterly_date is datasets_add_quarterly_date

    assert flat_make_lag_features is datasets_make_lag_features
    assert data_make_lag_features is datasets_make_lag_features

    assert flat_train_test_split_series is datasets_train_test_split_series
    assert data_train_test_split_series is datasets_train_test_split_series

    assert flat_load_raw_data is datasets_load_raw_data
    assert data_load_raw_data is datasets_load_raw_data


def test_build_parser_registers_core_commands():
    parser = build_parser()
    commands = parser._subparsers._group_actions[0].choices
    assert "describe" in commands
    assert "arima" in commands
    assert "lstm-train" in commands


def test_build_parser_parses_describe_command():
    parser = build_parser()
    args = parser.parse_args(["describe", "--state", "Maryland"])
    assert args.command == "describe"
    assert args.state == "Maryland"
