import pandas as pd

from inflation_forecasting.apps.arena import (
    available_model_catalog,
    build_leaderboard_frame,
    parse_order_text,
    prepare_arena_dataset,
    read_tabular_data,
)


def test_read_tabular_data_reads_csv_bytes():
    payload = b"date,value\n2020-01-01,1.0\n2020-02-01,2.0\n"
    df = read_tabular_data("series.csv", payload)
    assert list(df.columns) == ["date", "value"]
    assert len(df) == 2


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
