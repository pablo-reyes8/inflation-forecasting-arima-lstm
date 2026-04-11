from importlib.util import find_spec

import pandas as pd
import pytest

from inflation_forecasting.apps.arena import (
    available_model_catalog,
    build_leaderboard_frame,
    parse_order_text,
    prepare_arena_dataset,
)
from inflation_forecasting.datasets.io import read_tabular_data
from inflation_forecasting.models.econometrics.sarima import infer_seasonal_period


def test_read_tabular_data_reads_csv_bytes():
    payload = b"date,value\n2020-01-01,1.0\n2020-02-01,2.0\n"
    df = read_tabular_data("series.csv", payload)
    assert list(df.columns) == ["date", "value"]
    assert len(df) == 2


def test_read_tabular_data_rejects_unknown_extension():
    with pytest.raises(ValueError):
        read_tabular_data("series.txt", b"value\n1.0\n")


def test_prepare_arena_dataset_from_year_quarter_slice():
    df = pd.DataFrame(
        {
            "state": ["A"] * 12 + ["B"] * 12,
            "year": [2020] * 4 + [2021] * 4 + [2022] * 4 + [2020] * 4 + [2021] * 4 + [2022] * 4,
            "quarter": [1, 2, 3, 4] * 6,
            "pi": list(range(24)),
            "pi_t": list(range(100, 124)),
        }
    )
    dataset = prepare_arena_dataset(
        df,
        dataset_name="test",
        target_col="pi",
        year_col="year",
        quarter_col="quarter",
        entity_col="state",
        entity_value="A",
        exog_cols=["pi_t"],
    )
    assert dataset.name == "test"
    assert dataset.exog_cols == ("pi_t",)
    assert len(dataset.series) == 12
    assert dataset.metadata["n_exog"] == 1


def test_available_model_catalog_marks_arimax_unavailable_without_exog():
    catalog = available_model_catalog(has_exog=False)
    assert catalog["arimax"]["available"] is False
    assert "exogenous" in catalog["arimax"]["reason"].lower()


def test_infer_seasonal_period_falls_back_to_quarterly_month_pattern():
    index = pd.to_datetime(["2020-03-31", "2020-06-30", "2020-12-31", "2021-03-31"])
    series = pd.Series([1.0, 1.1, 1.2, 1.3], index=index)
    assert infer_seasonal_period(series) == 4


def test_parse_order_text_and_leaderboard_frame():
    order = parse_order_text("1,0,2", 3)
    assert order == (1, 0, 2)

    leaderboard = build_leaderboard_frame(
        [
            type(
                "Result",
                (),
                {
                    "model_key": "arima",
                    "label": "ARIMA",
                    "family": "Econometrics",
                    "status": "ok",
                    "duration_seconds": 0.5,
                    "error": None,
                    "validation_metrics": {"rmse": 0.2},
                    "test_metrics": {"rmse": 0.3},
                },
            )()
        ]
    )
    assert leaderboard.loc[0, "validation_rmse"] == 0.2
    assert leaderboard.loc[0, "test_rmse"] == 0.3


def test_api_health_and_catalog_endpoints():
    if find_spec("fastapi") is None or find_spec("httpx") is None:
        pytest.skip("fastapi/httpx is not installed")

    from fastapi.testclient import TestClient

    from inflation_forecasting.api.app import app

    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] in {"ok", "degraded"}
    assert "dependencies" in health.json()

    catalog = client.get("/catalog/dataset")
    assert catalog.status_code == 200
    assert catalog.json()["dataset"]["path"] == "Data/RawData.csv"


def test_api_openapi_includes_typed_forecast_schemas():
    if find_spec("fastapi") is None or find_spec("httpx") is None:
        pytest.skip("fastapi/httpx is not installed")

    from fastapi.testclient import TestClient

    from inflation_forecasting.api.app import app

    client = TestClient(app)
    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    payload = schema.json()
    components = payload["components"]["schemas"]
    assert "ForecastResponse" in components
    assert "VolatilityResponse" in components
    assert payload["paths"]["/forecast/arima"]["post"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith("ForecastResponse")


def test_api_states_endpoint_returns_repository_states():
    if find_spec("fastapi") is None or find_spec("httpx") is None:
        pytest.skip("fastapi/httpx is not installed")

    from fastapi.testclient import TestClient

    from inflation_forecasting.api.app import app

    client = TestClient(app)
    response = client.get("/catalog/states")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] >= 1
    assert "Maryland" in payload["states"]


def test_api_arima_endpoint_handles_optional_dependency_gracefully():
    if find_spec("fastapi") is None or find_spec("httpx") is None:
        pytest.skip("fastapi/httpx is not installed")

    from fastapi.testclient import TestClient

    from inflation_forecasting.api.app import app

    client = TestClient(app)
    response = client.post("/forecast/arima", json={"state": "Maryland", "forecast_steps": 2})
    if find_spec("statsmodels") is None:
        assert response.status_code == 503
        assert "statsmodels" in response.json()["detail"].lower()
    else:
        assert response.status_code == 200
        payload = response.json()
        assert payload["model"] == "arima"
        assert "series_summary" in payload
        assert len(payload["forecast"]) == 2


def test_api_garch_endpoint_handles_optional_dependency_gracefully():
    if find_spec("fastapi") is None or find_spec("httpx") is None:
        pytest.skip("fastapi/httpx is not installed")

    from fastapi.testclient import TestClient

    from inflation_forecasting.api.app import app

    client = TestClient(app)
    response = client.post("/volatility/garch", json={"state": "Maryland"})
    if find_spec("arch") is None:
        assert response.status_code == 503
        assert "arch" in response.json()["detail"].lower()
    else:
        assert response.status_code == 200
        payload = response.json()
        assert payload["model"] == "garch"
        assert len(payload["conditional_volatility"]) >= 1
