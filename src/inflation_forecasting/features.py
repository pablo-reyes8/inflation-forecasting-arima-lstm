"""Legacy flat import path.

Prefer `inflation_forecasting.datasets.features` for active code.
"""

from .datasets.features import make_lag_features

__all__ = ["make_lag_features"]
