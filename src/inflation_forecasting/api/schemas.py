from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class APIError(BaseModel):
    detail: str = Field(description="Human-readable error message.", examples=["statsmodels is required for ARIMA."])


class DependencyStatus(BaseModel):
    fastapi: bool = Field(description="Whether FastAPI is importable in the current environment.")
    statsmodels: bool = Field(description="Whether classical forecasting models are available.")
    arch: bool = Field(description="Whether ARCH/GARCH volatility models are available.")


class LinkSet(BaseModel):
    docs: str = Field(default="/docs")
    redoc: str = Field(default="/redoc")
    openapi: str = Field(default="/openapi.json")


class RootResponse(BaseModel):
    name: str
    status: Literal["ok"]
    version: str
    docs: LinkSet
    sections: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    dependencies: DependencyStatus


class TimeCoverage(BaseModel):
    start_year: int
    end_year: int
    frequency: str


class DatasetMetadataResponse(BaseModel):
    name: str
    path: str
    grain: str
    primary_key: list[str]
    time_coverage: TimeCoverage
    notes: list[str]


class DataDictionaryFieldResponse(BaseModel):
    name: str
    dtype: str
    role: str
    description: str
    nullable: bool


class DataAssetResponse(BaseModel):
    path: str
    role: str
    description: str


class DatasetCatalogResponse(BaseModel):
    dataset: DatasetMetadataResponse
    data_dictionary: list[DataDictionaryFieldResponse]
    assets: list[DataAssetResponse]


class ModelCatalogEntryResponse(BaseModel):
    label: str
    family: str
    requires: tuple[str, ...]
    available: bool
    reason: str | None = None
    needs_exog: bool = False


class ModelCatalogResponse(BaseModel):
    models: dict[str, ModelCatalogEntryResponse]


class StatesResponse(BaseModel):
    count: int
    states: list[str]


class SeriesSummary(BaseModel):
    observations: int
    missing_values: int
    start: str | None = None
    end: str | None = None
    frequency: str | None = None


class TimeSeriesPoint(BaseModel):
    date: str
    value: float | None = None


class SeriesRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "state": "Maryland",
                "target": "pi",
                "data_path": None,
                "year_col": "year",
                "quarter_col": "quarter",
                "date_col": "date",
                "interpolate": True,
            }
        }
    )

    data_path: str | None = Field(default=None, description="Optional path to a CSV/XLSX dataset.")
    state: str = Field(default="Maryland", description="Entity value to slice from the panel dataset.")
    target: str = Field(default="pi", description="Target column to forecast or analyze.")
    year_col: str = Field(default="year", description="Year column used to build a quarterly index.")
    quarter_col: str = Field(default="quarter", description="Quarter column used to build a quarterly index.")
    date_col: str = Field(default="date", description="Date column name used after preprocessing.")
    interpolate: bool = Field(default=True, description="Whether missing target values should be interpolated.")


class ArimaForecastRequest(SeriesRequest):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "state": "Maryland",
                "target": "pi",
                "order": [1, 0, 3],
                "test_size": 0.2,
                "forecast_steps": 4,
            }
        }
    )

    order: tuple[int, int, int] = Field(default=(1, 0, 3), description="ARIMA order as (p, d, q).")
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0, description="Fraction of observations reserved for evaluation.")
    forecast_steps: int = Field(default=4, ge=1, description="Number of future periods to forecast.")


class SarimaForecastRequest(SeriesRequest):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "state": "Maryland",
                "target": "pi",
                "order": [1, 0, 1],
                "seasonal_order": [1, 0, 1, 4],
                "auto_seasonal": True,
                "test_size": 0.2,
                "forecast_steps": 4,
            }
        }
    )

    order: tuple[int, int, int] = Field(default=(1, 0, 1), description="SARIMA non-seasonal order as (p, d, q).")
    seasonal_order: tuple[int, int, int, int] | None = Field(
        default=None,
        description="Optional SARIMA seasonal order as (P, D, Q, s).",
    )
    auto_seasonal: bool = Field(default=True, description="Infer the seasonal period when no seasonal_order is provided.")
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0, description="Fraction of observations reserved for evaluation.")
    forecast_steps: int = Field(default=4, ge=1, description="Number of future periods to forecast.")


class VolatilityRequest(SeriesRequest):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "state": "Maryland",
                "target": "pi",
                "p": 1,
                "q": 1,
                "mean": "constant",
            }
        }
    )

    p: int = Field(default=1, ge=1, description="Autoregressive volatility order.")
    q: int = Field(default=1, ge=1, description="Moving-average volatility order for GARCH.")
    mean: str = Field(default="constant", description="Mean process used by the volatility model.")


class ForecastResponse(BaseModel):
    model: Literal["arima", "sarima"]
    state: str
    target: str
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int] | None = None
    series_summary: SeriesSummary
    metrics: dict[str, float]
    test_predictions: list[TimeSeriesPoint]
    forecast: list[TimeSeriesPoint]


class VolatilityResponse(BaseModel):
    model: Literal["arch", "garch"]
    state: str
    target: str
    p: int
    q: int | None = None
    mean: str
    series_summary: SeriesSummary
    metrics: dict[str, float]
    conditional_volatility: list[TimeSeriesPoint]
