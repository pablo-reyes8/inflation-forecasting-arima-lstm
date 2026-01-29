import pandas as pd

from inflation_forecasting.preprocess import add_quarterly_date, prepare_state_series


def test_add_quarterly_date():
    df = pd.DataFrame({
        "state": ["A", "A"],
        "year": [2020, 2020],
        "quarter": [1, 2],
        "pi": [1.0, 2.0],
    })
    out = add_quarterly_date(df)
    assert "date" in out.columns
    assert out["date"].dtype.kind == "M"


def test_prepare_state_series():
    df = pd.DataFrame({
        "state": ["A", "A", "B"],
        "year": [2020, 2020, 2020],
        "quarter": [1, 2, 1],
        "pi": [1.0, None, 3.0],
    })
    series = prepare_state_series(df, state="A", interpolate=True)
    assert series.isna().sum() == 0
    assert len(series) == 2
