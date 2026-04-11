from __future__ import annotations

from importlib.util import find_spec

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from .. import __version__
from ..apps.arena import available_model_catalog
from ..dataops.catalog import DATA_ASSETS, RAW_DATA_DICTIONARY, RAW_DATASET_METADATA
from ..datasets.io import load_raw_data
from ..datasets.preprocessing import prepare_state_series
from ..models.econometrics.arch_garch import fit_arch, fit_garch
from ..models.econometrics.arima import evaluate_arima, fit_arima, forecast_arima
from ..models.econometrics.sarima import evaluate_sarima, fit_sarima, infer_seasonal_period
from .schemas import (
    APIError,
    ArimaForecastRequest,
    DatasetCatalogResponse,
    DependencyStatus,
    ForecastResponse,
    HealthResponse,
    LinkSet,
    ModelCatalogResponse,
    RootResponse,
    SarimaForecastRequest,
    SeriesRequest,
    SeriesSummary,
    StatesResponse,
    TimeSeriesPoint,
    VolatilityRequest,
    VolatilityResponse,
)


TAGS_METADATA = [
    {"name": "Meta", "description": "Health checks, entrypoints, and runtime capability signals."},
    {"name": "Catalog", "description": "Dataset and model metadata exposed for clients and dashboards."},
    {"name": "Forecasting", "description": "Classical time-series forecast endpoints over the canonical panel."},
    {"name": "Volatility", "description": "ARCH and GARCH volatility estimation endpoints."},
]

COMMON_OPERATION_ERRORS = {
    400: {"model": APIError, "description": "Invalid request or dataset slice."},
    404: {"model": APIError, "description": "Referenced dataset path does not exist."},
    503: {"model": APIError, "description": "Optional model dependency is not installed."},
}


def _series_payload(series: pd.Series) -> list[TimeSeriesPoint]:
    return [
        TimeSeriesPoint(
            date=index.isoformat() if hasattr(index, "isoformat") else str(index),
            value=None if value is None else float(value),
        )
        for index, value in series.items()
    ]


def _series_summary(series: pd.Series) -> SeriesSummary:
    clean = series.dropna()
    frequency = None
    if len(clean.index) >= 3:
        frequency = pd.infer_freq(clean.index)
    return SeriesSummary(
        observations=int(len(series)),
        missing_values=int(series.isna().sum()),
        start=clean.index.min().isoformat() if not clean.empty else None,
        end=clean.index.max().isoformat() if not clean.empty else None,
        frequency=frequency,
    )


def _load_series(request: SeriesRequest) -> pd.Series:
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


def _dependency_status() -> DependencyStatus:
    return DependencyStatus(
        fastapi=find_spec("fastapi") is not None,
        statsmodels=find_spec("statsmodels") is not None,
        arch=find_spec("arch") is not None,
    )


def _translate_exception(exc: Exception) -> HTTPException:
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ImportError):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail="Unexpected server error.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Inflation Forecasting API",
        version="0.3.0",
        summary="Typed forecasting API for state-level inflation research workflows.",
        description=(
            "A documented HTTP layer for dataset inspection, benchmark forecasting, and volatility modeling.\n\n"
            "The API is designed for analytical notebooks, dashboards, and lightweight product integrations. "
            "Each endpoint exposes explicit schemas, request examples, and dependency-aware error responses."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_tags=TAGS_METADATA,
        contact={"name": "Inflation Forecasting Maintainers"},
        license_info={"name": "MIT"},
    )

    @app.get("/", response_model=RootResponse, tags=["Meta"], summary="API entrypoint")
    def root() -> RootResponse:
        return RootResponse(
            name="inflation-forecasting-api",
            status="ok",
            version=__version__,
            docs=LinkSet(),
            sections=["Meta", "Catalog", "Forecasting", "Volatility"],
        )

    @app.get("/health", response_model=HealthResponse, tags=["Meta"], summary="Service health and capabilities")
    def health() -> HealthResponse:
        dependencies = _dependency_status()
        status = "ok" if dependencies.statsmodels and dependencies.arch else "degraded"
        return HealthResponse(status=status, dependencies=dependencies)

    @app.get(
        "/catalog/dataset",
        response_model=DatasetCatalogResponse,
        tags=["Catalog"],
        summary="Dataset contract and asset catalog",
    )
    def dataset_catalog() -> DatasetCatalogResponse:
        return DatasetCatalogResponse(
            dataset=RAW_DATASET_METADATA,
            data_dictionary=RAW_DATA_DICTIONARY,
            assets=DATA_ASSETS,
        )

    @app.get(
        "/catalog/models",
        response_model=ModelCatalogResponse,
        tags=["Catalog"],
        summary="Model availability matrix",
        description="Return the catalog of benchmark models together with dependency-driven availability flags.",
    )
    def model_catalog() -> ModelCatalogResponse:
        return ModelCatalogResponse(models=available_model_catalog(has_exog=True))

    @app.get(
        "/catalog/states",
        response_model=StatesResponse,
        tags=["Catalog"],
        summary="Available entities in the dataset",
        responses={404: COMMON_OPERATION_ERRORS[404]},
    )
    def states(
        data_path: str | None = Query(default=None, description="Optional path to a CSV/XLSX dataset to inspect.")
    ) -> StatesResponse:
        try:
            df = load_raw_data(data_path)
        except Exception as exc:  # pragma: no cover - unified API surface
            raise _translate_exception(exc) from exc
        return StatesResponse(count=int(df["state"].nunique()), states=sorted(df["state"].unique().tolist()))

    @app.post(
        "/forecast/arima",
        response_model=ForecastResponse,
        response_model_exclude_none=True,
        tags=["Forecasting"],
        summary="ARIMA benchmark forecast",
        responses=COMMON_OPERATION_ERRORS,
    )
    def arima_forecast(request: ArimaForecastRequest) -> ForecastResponse:
        try:
            series = _load_series(request)
            evaluation = evaluate_arima(series, order=request.order, test_size=request.test_size)
            model_fit = fit_arima(series, order=request.order)
            forecast = forecast_arima(model_fit, steps=request.forecast_steps)
        except Exception as exc:  # pragma: no cover - unified API surface
            raise _translate_exception(exc) from exc

        return ForecastResponse(
            model="arima",
            state=request.state,
            target=request.target,
            order=request.order,
            series_summary=_series_summary(series),
            metrics=evaluation.metrics,
            test_predictions=_series_payload(evaluation.predictions),
            forecast=_series_payload(forecast),
        )

    @app.post(
        "/forecast/sarima",
        response_model=ForecastResponse,
        response_model_exclude_none=True,
        tags=["Forecasting"],
        summary="SARIMA benchmark forecast",
        responses=COMMON_OPERATION_ERRORS,
    )
    def sarima_forecast(request: SarimaForecastRequest) -> ForecastResponse:
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
        except Exception as exc:  # pragma: no cover - unified API surface
            raise _translate_exception(exc) from exc

        return ForecastResponse(
            model="sarima",
            state=request.state,
            target=request.target,
            order=request.order,
            seasonal_order=seasonal_order,
            series_summary=_series_summary(series),
            metrics=evaluation.metrics,
            test_predictions=_series_payload(evaluation.predictions),
            forecast=_series_payload(forecast),
        )

    @app.post(
        "/volatility/arch",
        response_model=VolatilityResponse,
        response_model_exclude_none=True,
        tags=["Volatility"],
        summary="ARCH conditional volatility fit",
        responses=COMMON_OPERATION_ERRORS,
    )
    def arch_volatility(request: VolatilityRequest) -> VolatilityResponse:
        try:
            series = _load_series(request)
            result = fit_arch(series, p=request.p, mean=request.mean)
        except Exception as exc:  # pragma: no cover - unified API surface
            raise _translate_exception(exc) from exc

        return VolatilityResponse(
            model="arch",
            state=request.state,
            target=request.target,
            p=request.p,
            mean=request.mean,
            series_summary=_series_summary(series),
            metrics={"aic": result.aic, "bic": result.bic},
            conditional_volatility=_series_payload(result.conditional_volatility),
        )

    @app.post(
        "/volatility/garch",
        response_model=VolatilityResponse,
        response_model_exclude_none=True,
        tags=["Volatility"],
        summary="GARCH conditional volatility fit",
        responses=COMMON_OPERATION_ERRORS,
    )
    def garch_volatility(request: VolatilityRequest) -> VolatilityResponse:
        try:
            series = _load_series(request)
            result = fit_garch(series, p=request.p, q=request.q, mean=request.mean)
        except Exception as exc:  # pragma: no cover - unified API surface
            raise _translate_exception(exc) from exc

        return VolatilityResponse(
            model="garch",
            state=request.state,
            target=request.target,
            p=request.p,
            q=request.q,
            mean=request.mean,
            series_summary=_series_summary(series),
            metrics={"aic": result.aic, "bic": result.bic},
            conditional_volatility=_series_payload(result.conditional_volatility),
        )

    return app


def dependency_report() -> dict[str, bool]:
    status = _dependency_status()
    return status.model_dump()


app = create_app()
