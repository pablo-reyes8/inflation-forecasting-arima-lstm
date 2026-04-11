from __future__ import annotations

import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path


def main() -> int:
    if find_spec("streamlit") is None:
        raise ImportError("streamlit is required to run the arena app. Install with `pip install streamlit plotly`.")
    app_path = Path(__file__).resolve().parents[3] / "streamlit_app.py"
    command = [sys.executable, "-m", "streamlit", "run", str(app_path), *sys.argv[1:]]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
