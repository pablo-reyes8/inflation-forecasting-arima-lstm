from __future__ import annotations

from typing import Optional

import pandas as pd


def plot_series(series: pd.Series, title: str = "Series", output_path: Optional[str] = None) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting. Install with `pip install matplotlib`.") from exc

    plt.figure(figsize=(12, 6))
    plt.plot(series.index, series.values, linewidth=2)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(series.name or "value")
    plt.grid(True)
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
    else:
        plt.show()
