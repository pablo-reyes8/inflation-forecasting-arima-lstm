from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from inflation_forecasting.apps.streamlit_arena import main
except ImportError as exc:
    if getattr(exc, "name", None) == "streamlit":
        raise ImportError("streamlit is required to run the arena app. Install with `pip install streamlit plotly`.") from exc
    raise


if __name__ == "__main__":
    main()
