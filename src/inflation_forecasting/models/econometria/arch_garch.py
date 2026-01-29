from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def _require_arch():
    try:
        from arch import arch_model  # noqa: F401
    except ImportError as exc:
        raise ImportError("arch is required for ARCH/GARCH. Install with `pip install arch`." ) from exc


@dataclass
class VolatilityResult:
    model_fit: object
    aic: float
    bic: float
    conditional_volatility: pd.Series


def fit_arch(series: pd.Series, p: int = 1, mean: str = "constant") -> VolatilityResult:
    _require_arch()
    from arch import arch_model

    model = arch_model(series, mean=mean, vol="ARCH", p=p, rescale=False)
    res = model.fit(disp="off")
    vol = pd.Series(res.conditional_volatility, index=series.index, name="cond_vol")
    return VolatilityResult(model_fit=res, aic=res.aic, bic=res.bic, conditional_volatility=vol)


def fit_garch(series: pd.Series, p: int = 1, q: int = 1, mean: str = "constant") -> VolatilityResult:
    _require_arch()
    from arch import arch_model

    model = arch_model(series, mean=mean, vol="GARCH", p=p, q=q, rescale=False)
    res = model.fit(disp="off")
    vol = pd.Series(res.conditional_volatility, index=series.index, name="cond_vol")
    return VolatilityResult(model_fit=res, aic=res.aic, bic=res.bic, conditional_volatility=vol)
