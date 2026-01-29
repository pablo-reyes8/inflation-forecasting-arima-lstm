from __future__ import annotations

from typing import Optional

import pandas as pd


def add_quarterly_date(
    df: pd.DataFrame,
    year_col: str = "year",
    quarter_col: str = "quarter",
    date_col: str = "date",
) -> pd.DataFrame:
    df = df.copy()
    qstrings = df[year_col].astype(str) + "Q" + df[quarter_col].astype(str)
    dates = pd.PeriodIndex(qstrings, freq="Q").to_timestamp(how="end")
    df[date_col] = dates
    return df


def filter_state(
    df: pd.DataFrame,
    state: str,
    state_col: str = "state",
) -> pd.DataFrame:
    return df.loc[df[state_col] == state].copy()


def to_time_series(
    df: pd.DataFrame,
    target: str = "pi",
    date_col: str = "date",
    sort: bool = True,
) -> pd.Series:
    series = df.set_index(date_col)[target]
    if sort:
        series = series.sort_index()
    return series


def interpolate_series(series: pd.Series, method: str = "linear") -> pd.Series:
    return series.interpolate(method=method)


def prepare_state_series(
    df: pd.DataFrame,
    state: str = "Maryland",
    target: str = "pi",
    year_col: str = "year",
    quarter_col: str = "quarter",
    date_col: str = "date",
    interpolate: bool = True,
) -> pd.Series:
    df = add_quarterly_date(df, year_col=year_col, quarter_col=quarter_col, date_col=date_col)
    df_state = filter_state(df, state=state)
    series = to_time_series(df_state, target=target, date_col=date_col)
    if interpolate:
        series = interpolate_series(series)
    return series


def summary_stats(series: pd.Series) -> pd.DataFrame:
    stats = {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "var": series.var(),
        "min": series.min(),
        "max": series.max(),
        "count": series.count(),
    }
    return pd.DataFrame(stats, index=[series.name or "value"])
