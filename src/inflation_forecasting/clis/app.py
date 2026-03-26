from __future__ import annotations

import argparse

from .. import __version__
from .common import add_common_args
from .data import register_data_commands
from .econometrics import register_econometrics_commands
from .ml import register_ml_commands
from .neural import register_neural_commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inflation-forecast", description="Inflation forecasting toolkit")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    common = argparse.ArgumentParser(add_help=False)
    add_common_args(common)

    subparsers = parser.add_subparsers(dest="command", required=True)
    register_data_commands(subparsers, common)
    register_econometrics_commands(subparsers, common)
    register_neural_commands(subparsers, common)
    register_ml_commands(subparsers, common)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
