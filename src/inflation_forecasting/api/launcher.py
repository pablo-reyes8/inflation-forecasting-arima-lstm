from __future__ import annotations

import os
from importlib.util import find_spec


def main() -> int:
    missing = [name for name in ("fastapi", "uvicorn") if find_spec(name) is None]
    if missing:
        raise ImportError(
            "The API requires these packages: "
            + ", ".join(missing)
            + ". Install with `pip install fastapi uvicorn`."
        )

    import uvicorn

    host = os.getenv("INFLATION_FORECAST_API_HOST", "127.0.0.1")
    port = int(os.getenv("INFLATION_FORECAST_API_PORT", "8000"))
    uvicorn.run("inflation_forecasting.api.app:app", host=host, port=port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
