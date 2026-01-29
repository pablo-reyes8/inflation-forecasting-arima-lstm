from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return project_root() / "Data"


def outputs_dir() -> Path:
    return project_root() / "outputs"
