import pandas as pd
import pytest

from inflation_forecasting.datasets.preprocessing import add_quarterly_date, filter_state, prepare_state_series, summary_stats


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


def test_add_quarterly_date_rejects_invalid_quarter():
    df = pd.DataFrame({"year": [2020], "quarter": [5]})
    with pytest.raises(ValueError):
        add_quarterly_date(df)


def test_filter_state_raises_when_missing():
    df = pd.DataFrame({"state": ["A"], "pi": [1.0]})
    with pytest.raises(ValueError):
        filter_state(df, state="B")


def test_summary_stats_reports_missing_values():
    series = pd.Series([1.0, None, 3.0], index=pd.date_range("2020-03-31", periods=3, freq="QE-DEC"), name="pi")
    stats = summary_stats(series)
    assert stats.loc["pi", "missing"] == 1
