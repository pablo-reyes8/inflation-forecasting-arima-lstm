from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


RAW_DATA_REQUIRED_COLUMNS = ("state", "year", "quarter", "pi_nt", "pi_t", "pi")


def validate_required_columns(df: pd.DataFrame, required_columns: Sequence[str]) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def add_quarterly_date(
    df: pd.DataFrame,
    year_col: str = "year",
    quarter_col: str = "quarter",
    date_col: str = "date",
) -> pd.DataFrame:
    validate_required_columns(df, (year_col, quarter_col))
    if df[[year_col, quarter_col]].isna().any().any():
        raise ValueError(f"Columns '{year_col}' and '{quarter_col}' must not contain missing values.")

    invalid_quarters = sorted(df.loc[~df[quarter_col].isin([1, 2, 3, 4]), quarter_col].unique().tolist())
    if invalid_quarters:
        raise ValueError(f"Quarter column '{quarter_col}' contains invalid values: {invalid_quarters}")

    df = df.copy()
    qstrings = df[year_col].astype(int).astype(str) + "Q" + df[quarter_col].astype(int).astype(str)
    dates = pd.PeriodIndex(qstrings, freq="Q").to_timestamp(how="end").normalize()
    df[date_col] = dates
    return df


def filter_state(
    df: pd.DataFrame,
    state: str,
    state_col: str = "state",
    allow_empty: bool = False,
) -> pd.DataFrame:
    validate_required_columns(df, (state_col,))
    filtered = df.loc[df[state_col] == state].copy()
    if filtered.empty and not allow_empty:
        raise ValueError(f"State '{state}' was not found in column '{state_col}'.")
    return filtered


def to_time_series(
    df: pd.DataFrame,
    target: str = "pi",
    date_col: str = "date",
    sort: bool = True,
) -> pd.Series:
    validate_required_columns(df, (date_col, target))
    if df[date_col].duplicated().any():
        duplicates = df.loc[df[date_col].duplicated(), date_col].astype(str).tolist()
        raise ValueError(f"Duplicate dates found in column '{date_col}': {duplicates[:5]}")

    series = df.set_index(date_col)[target]
    if sort:
        series = series.sort_index()
    series.name = target
    return series


def interpolate_series(series: pd.Series, method: str = "linear") -> pd.Series:
    return series.astype(float).interpolate(method=method)


def prepare_state_series(
    df: pd.DataFrame,
    state: str = "Maryland",
    target: str = "pi",
    year_col: str = "year",
    quarter_col: str = "quarter",
    date_col: str = "date",
    state_col: str = "state",
    interpolate: bool = True,
) -> pd.Series:
    df = add_quarterly_date(df, year_col=year_col, quarter_col=quarter_col, date_col=date_col)
    df_state = filter_state(df, state=state, state_col=state_col)
    series = to_time_series(df_state, target=target, date_col=date_col)
    if interpolate:
        series = interpolate_series(series)
    return series


def summary_stats(series: pd.Series) -> pd.DataFrame:
    clean = series.dropna()
    frequency = None
    if len(clean.index) >= 3:
        frequency = pd.infer_freq(clean.index)

    stats = {
        "mean": clean.mean(),
        "median": clean.median(),
        "std": clean.std(),
        "var": clean.var(),
        "min": clean.min(),
        "max": clean.max(),
        "count": int(clean.count()),
        "missing": int(series.isna().sum()),
        "start": clean.index.min() if not clean.empty else None,
        "end": clean.index.max() if not clean.empty else None,
        "frequency": frequency,
    }
    return pd.DataFrame(stats, index=[series.name or "value"])
