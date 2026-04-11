from __future__ import annotations

from importlib.util import find_spec

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..apps.arena import available_model_catalog
from ..dataops.catalog import DATA_ASSETS, RAW_DATA_DICTIONARY, RAW_DATASET_METADATA
from ..datasets.io import load_raw_data
from ..datasets.preprocessing import prepare_state_series
from ..models.econometrics.arch_garch import fit_arch, fit_garch
from ..models.econometrics.arima import evaluate_arima, fit_arima, forecast_arima
from ..models.econometrics.sarima import evaluate_sarima, fit_sarima, infer_seasonal_period


def _series_payload(series) -> list[dict[str, float | str | None]]:
    return [
        {
            "date": index.isoformat() if hasattr(index, "isoformat") else str(index),
            "value": None if value is None else float(value),
        }
        for index, value in series.items()
    ]


def _load_series(request: "SeriesRequest"):
    df = load_raw_data(request.data_path)
    return prepare_state_series(
        df,
        state=request.state,
        target=request.target,
        year_col=request.year_col,
        quarter_col=request.quarter_col,
        date_col=request.date_col,
        interpolate=request.interpolate,
    )


class SeriesRequest(BaseModel):
    data_path: str | None = Field(default=None, description="Optional path to a CSV/XLSX dataset.")
    state: str = Field(default="Maryland")
    target: str = Field(default="pi")
    year_col: str = Field(default="year")
    quarter_col: str = Field(default="quarter")
    date_col: str = Field(default="date")
    interpolate: bool = Field(default=True)


class ArimaForecastRequest(SeriesRequest):
    order: tuple[int, int, int] = Field(default=(1, 0, 3))
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    forecast_steps: int = Field(default=4, ge=1)


class SarimaForecastRequest(SeriesRequest):
    order: tuple[int, int, int] = Field(default=(1, 0, 1))
    seasonal_order: tuple[int, int, int, int] | None = Field(default=None)
    auto_seasonal: bool = Field(default=True)
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    forecast_steps: int = Field(default=4, ge=1)


class VolatilityRequest(SeriesRequest):
    p: int = Field(default=1, ge=1)
    q: int = Field(default=1, ge=1)
    mean: str = Field(default="constant")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Inflation Forecasting API",
        version="0.3.0",
        description="HTTP interface for dataset inspection, classical forecasts and volatility models.",
    )

    @app.get("/")
    def root() -> dict[str, str]:
        return {
            "name": "inflation-forecasting-api",
            "status": "ok",
            "docs": "/docs",
        }

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/catalog/dataset")
    def dataset_catalog() -> dict[str, object]:
        return {
            "dataset": RAW_DATASET_METADATA,
            "data_dictionary": RAW_DATA_DICTIONARY,
            "assets": DATA_ASSETS,
        }

    @app.get("/catalog/models")
    def model_catalog() -> dict[str, object]:
        return available_model_catalog(has_exog=True)

    @app.get("/catalog/states")
    def states(data_path: str | None = None) -> dict[str, object]:
        df = load_raw_data(data_path)
        return {
            "count": int(df["state"].nunique()),
            "states": sorted(df["state"].unique().tolist()),
        }

    @app.post("/forecast/arima")
    def arima_forecast(request: ArimaForecastRequest) -> dict[str, object]:
        try:
            series = _load_series(request)
            evaluation = evaluate_arima(series, order=request.order, test_size=request.test_size)
            model_fit = fit_arima(series, order=request.order)
            forecast = forecast_arima(model_fit, steps=request.forecast_steps)
        except ImportError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {
            "model": "arima",
            "state": request.state,
            "target": request.target,
            "order": request.order,
            "metrics": evaluation.metrics,
            "test_predictions": _series_payload(evaluation.predictions),
            "forecast": _series_payload(forecast),
        }

    @app.post("/forecast/sarima")
    def sarima_forecast(request: SarimaForecastRequest) -> dict[str, object]:
        try:
            series = _load_series(request)
            seasonal_order = request.seasonal_order
            if seasonal_order is None and request.auto_seasonal:
                period = infer_seasonal_period(series)
                seasonal_order = (1, 0, 1, period) if period else (0, 0, 0, 0)
            elif seasonal_order is None:
                seasonal_order = (0, 0, 0, 0)
            evaluation = evaluate_sarima(series, order=request.order, seasonal_order=seasonal_order, test_size=request.test_size)
            model_fit = fit_sarima(series, order=request.order, seasonal_order=seasonal_order)
            forecast = model_fit.forecast(steps=request.forecast_steps)
        except ImportError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {
            "model": "sarima",
            "state": request.state,
            "target": request.target,
            "order": request.order,
            "seasonal_order": seasonal_order,
            "metrics": evaluation.metrics,
            "test_predictions": _series_payload(evaluation.predictions),
            "forecast": _series_payload(forecast),
        }

    @app.post("/volatility/arch")
    def arch_volatility(request: VolatilityRequest) -> dict[str, object]:
        try:
            series = _load_series(request)
            result = fit_arch(series, p=request.p, mean=request.mean)
        except ImportError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {
            "model": "arch",
            "state": request.state,
            "target": request.target,
            "p": request.p,
            "mean": request.mean,
            "metrics": {"aic": result.aic, "bic": result.bic},
            "conditional_volatility": _series_payload(result.conditional_volatility),
        }

    @app.post("/volatility/garch")
    def garch_volatility(request: VolatilityRequest) -> dict[str, object]:
        try:
            series = _load_series(request)
            result = fit_garch(series, p=request.p, q=request.q, mean=request.mean)
        except ImportError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {
            "model": "garch",
            "state": request.state,
            "target": request.target,
            "p": request.p,
            "q": request.q,
            "mean": request.mean,
            "metrics": {"aic": result.aic, "bic": result.bic},
            "conditional_volatility": _series_payload(result.conditional_volatility),
        }

    return app


def dependency_report() -> dict[str, bool]:
    return {
        "fastapi": find_spec("fastapi") is not None,
        "statsmodels": find_spec("statsmodels") is not None,
        "arch": find_spec("arch") is not None,
    }


app = create_app()
