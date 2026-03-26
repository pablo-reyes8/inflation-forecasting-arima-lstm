from __future__ import annotations

import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .. import __version__


def _normalize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def build_series_metadata(
    series: pd.Series | None,
    *,
    data_source: str | Path | None,
    state: str | None,
    target: str | None,
) -> dict[str, Any]:
    if series is None:
        return {"source_path": str(data_source) if data_source else None, "state": state, "target": target}

    clean = series.dropna()
    inferred_frequency = None
    if len(clean.index) >= 3:
        inferred_frequency = pd.infer_freq(clean.index)

    return {
        "source_path": str(data_source) if data_source else None,
        "state": state,
        "target": target,
        "n_observations": int(len(series)),
        "n_missing": int(series.isna().sum()),
        "date_start": clean.index.min() if not clean.empty else None,
        "date_end": clean.index.max() if not clean.empty else None,
        "frequency": inferred_frequency,
    }


def build_run_manifest(
    *,
    run_id: str,
    command_name: str,
    model_family: str,
    model_name: str,
    parameters: Mapping[str, Any],
    metrics: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, Any] | None = None,
    series: pd.Series | None = None,
    data_source: str | Path | None = None,
    state: str | None = None,
    target: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "run": {
            "id": run_id,
            "command": command_name,
            "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "package_version": __version__,
        },
        "data": build_series_metadata(series, data_source=data_source, state=state, target=target),
        "model": {
            "family": model_family,
            "name": model_name,
            "parameters": _normalize(dict(parameters)),
        },
        "metrics": _normalize(dict(metrics or {})),
        "artifacts": _normalize(dict(artifacts or {})),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    if extra:
        manifest["extra"] = _normalize(dict(extra))
    return manifest
