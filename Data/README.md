# Data Assets

This directory stores the canonical dataset used by the package plus legacy spreadsheet exports preserved for traceability.

The repository code that loads and preprocesses these assets lives in `src/inflation_forecasting/datasets/`, intentionally named to avoid confusion with this `Data/` folder.

## Canonical Dataset

- File: `RawData.csv`
- Shape: `4699` rows, `6` columns
- Grain: one observation per `state-year-quarter`
- Coverage: `1978Q1` to `2017Q4`
- Entities: `34` state-level series including the District of Columbia
- Primary key: `state`, `year`, `quarter`
- Default target for forecasting: `pi`

## Files

| File | Role | Notes |
|------|------|-------|
| `RawData.csv` | Canonical raw panel | Source used by the Python package and CLI. |
| `Data Cleaned.xlsx` | Legacy spreadsheet | Preserved from the original academic workflow. |
| `ARIMA Forecasting and Errors.xlsx` | Legacy results export | Historical ARIMA outputs kept for reference. |

## Column Summary

| Column | Type | Role | Description |
|--------|------|------|-------------|
| `state` | string | Entity identifier | State or district name. |
| `year` | int64 | Time key | Calendar year of the observation. |
| `quarter` | int64 | Time key | Quarter index with valid domain `1-4`. |
| `pi_nt` | float64 | Feature | Inflation rate for non-tradable goods. |
| `pi_t` | float64 | Feature | Inflation rate for tradable goods. |
| `pi` | float64 | Target | Headline inflation rate. |

The machine-readable column contract is stored in `data_dictionary.yml`.
