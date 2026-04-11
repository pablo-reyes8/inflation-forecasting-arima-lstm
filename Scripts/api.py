from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inflation_forecasting.api.launcher import main


if __name__ == "__main__":
    raise SystemExit(main())
