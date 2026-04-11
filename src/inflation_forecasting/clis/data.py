from __future__ import annotations

import argparse

from ..datasets.io import load_raw_data
from ..datasets.preprocessing import summary_stats
from ..dataops.quality import assert_quality_gate, audit_inflation_dataset
from ..decomposition import hp_filter, seasonal_decompose_series
from .common import create_run, load_series, save_json_artifact, save_table_artifact, save_yaml_artifact, write_manifest


def cmd_describe(args: argparse.Namespace) -> None:
    series = load_series(args)
    stats = summary_stats(series)
    run = create_run(args, "describe")
    artifact_paths = {"summary": save_table_artifact(run, "summary", stats)}
    artifact_paths["manifest"] = write_manifest(
        args=args,
        run=run,
        command_name="describe",
        model_family="data",
        model_name="summary_statistics",
        series=series,
        metrics=None,
        artifact_paths=artifact_paths,
    )
    print(stats.to_string())
    print(f"Saved run artifacts to {run.path}")


def cmd_hp_filter(args: argparse.Namespace) -> None:
    series = load_series(args)
    trend, cycle = hp_filter(series, lamb=args.lamb)
    output = trend.to_frame("trend")
    output["cycle"] = cycle
    run = create_run(args, "hp-filter")
    artifact_paths = {"trend_cycle": save_table_artifact(run, "trend_cycle", output)}
    artifact_paths["manifest"] = write_manifest(
        args=args,
        run=run,
        command_name="hp-filter",
        model_family="decomposition",
        model_name="hp_filter",
        series=series,
        metrics={"lambda": args.lamb},
        artifact_paths=artifact_paths,
    )
    print(f"Saved run artifacts to {run.path}")


def cmd_decompose(args: argparse.Namespace) -> None:
    series = load_series(args)
    result = seasonal_decompose_series(series, model=args.model, period=args.period)
    output = result.trend.to_frame("trend")
    output["seasonal"] = result.seasonal
    output["resid"] = result.resid
    run = create_run(args, "decompose")
    artifact_paths = {"decomposition": save_table_artifact(run, "decomposition", output)}
    artifact_paths["manifest"] = write_manifest(
        args=args,
        run=run,
        command_name="decompose",
        model_family="decomposition",
        model_name=args.model,
        series=series,
        metrics={"period": args.period},
        artifact_paths=artifact_paths,
    )
    print(f"Saved run artifacts to {run.path}")


def cmd_data_audit(args: argparse.Namespace) -> None:
    df = load_raw_data(args.data_path)
    report = audit_inflation_dataset(df)
    if args.strict:
        assert_quality_gate(report)

    run = create_run(args, "data-audit")
    payload = report.to_dict()
    artifact_paths = {
        "quality_report": save_json_artifact(run, "quality_report", payload),
        "quality_report_manifest": save_yaml_artifact(run, "quality_report", payload),
    }
    artifact_paths["manifest"] = write_manifest(
        args=args,
        run=run,
        command_name="data-audit",
        model_family="dataops",
        model_name="quality_audit",
        series=None,
        metrics={"issue_count": len(report.issues)},
        artifact_paths=artifact_paths,
        extra={"issues": report.issues},
    )
    print(payload)
    print(f"Saved run artifacts to {run.path}")


def register_data_commands(subparsers, common: argparse.ArgumentParser) -> None:
    describe = subparsers.add_parser("describe", parents=[common], help="Descriptive stats")
    describe.set_defaults(func=cmd_describe)

    hp = subparsers.add_parser("hp-filter", parents=[common], help="HP filter trend/cycle")
    hp.add_argument("--lamb", type=float, default=1600.0)
    hp.set_defaults(func=cmd_hp_filter)

    decompose = subparsers.add_parser("decompose", parents=[common], help="Seasonal decomposition")
    decompose.add_argument("--model", default="additive")
    decompose.add_argument("--period", type=int, default=4)
    decompose.set_defaults(func=cmd_decompose)

    audit = subparsers.add_parser("data-audit", parents=[common], help="Run dataset quality checks")
    audit.add_argument("--strict", action="store_true")
    audit.set_defaults(func=cmd_data_audit)
