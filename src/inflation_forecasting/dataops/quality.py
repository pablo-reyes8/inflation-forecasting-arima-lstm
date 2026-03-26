from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from ..data.preprocessing import RAW_DATA_REQUIRED_COLUMNS, add_quarterly_date, validate_required_columns


@dataclass
class DatasetQualityReport:
    rows: int
    columns: int
    states: int
    year_min: int | None
    year_max: int | None
    min_rows_per_state: int | None
    max_rows_per_state: int | None
    duplicates_on_primary_key: int
    invalid_quarters: int
    missing_by_column: dict[str, int]
    inferred_frequency: str | None
    issues: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def audit_inflation_dataset(
    df: pd.DataFrame,
    *,
    state_col: str = "state",
    year_col: str = "year",
    quarter_col: str = "quarter",
    primary_key: tuple[str, str, str] = ("state", "year", "quarter"),
) -> DatasetQualityReport:
    validate_required_columns(df, RAW_DATA_REQUIRED_COLUMNS)

    missing_by_column = {column: int(count) for column, count in df.isna().sum().items()}
    duplicates = int(df.duplicated(list(primary_key)).sum())
    invalid_quarters = int((~df[quarter_col].isin([1, 2, 3, 4])).sum())
    grouped = df.groupby(state_col).size() if state_col in df.columns else pd.Series(dtype=int)

    inferred_frequency = None
    issues: list[str] = []
    if invalid_quarters == 0:
        dated = add_quarterly_date(df, year_col=year_col, quarter_col=quarter_col, date_col="date")
        sample_state = dated[state_col].iloc[0]
        sample_series = dated.loc[dated[state_col] == sample_state, "date"].sort_values()
        if len(sample_series) >= 3:
            inferred_frequency = pd.infer_freq(sample_series)

    if duplicates:
        issues.append(f"Found {duplicates} duplicate rows on primary key {list(primary_key)}.")
    if invalid_quarters:
        issues.append(f"Found {invalid_quarters} rows with invalid quarter values.")

    missing_columns = [column for column, count in missing_by_column.items() if count > 0]
    if missing_columns:
        issues.append(f"Missing values detected in columns: {missing_columns}.")

    return DatasetQualityReport(
        rows=int(len(df)),
        columns=int(len(df.columns)),
        states=int(df[state_col].nunique()),
        year_min=int(df[year_col].min()) if not df.empty else None,
        year_max=int(df[year_col].max()) if not df.empty else None,
        min_rows_per_state=int(grouped.min()) if not grouped.empty else None,
        max_rows_per_state=int(grouped.max()) if not grouped.empty else None,
        duplicates_on_primary_key=duplicates,
        invalid_quarters=invalid_quarters,
        missing_by_column=missing_by_column,
        inferred_frequency=inferred_frequency,
        issues=issues,
    )


def assert_quality_gate(report: DatasetQualityReport) -> None:
    if report.issues:
        raise ValueError("Dataset quality gate failed: " + " ".join(report.issues))
