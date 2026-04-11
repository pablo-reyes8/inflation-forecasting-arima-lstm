from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re

from ..datasets.io import default_outputs_dir, ensure_dir


def _slugify(value: str) -> str:
    compact = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return compact or "run"


@dataclass(frozen=True)
class RunDirectory:
    run_id: str
    path: Path


def create_run_directory(command_name: str, output_dir: str | Path | None = None) -> RunDirectory:
    root = Path(output_dir) if output_dir else default_outputs_dir()
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{_slugify(command_name)}"
    run_dir = ensure_dir(root / "runs" / run_id)
    return RunDirectory(run_id=run_id, path=run_dir)


def relative_artifact_path(path: str | Path, run_dir: str | Path) -> str:
    artifact_path = Path(path)
    base_dir = Path(run_dir)
    try:
        return str(artifact_path.relative_to(base_dir))
    except ValueError:
        return str(artifact_path)
