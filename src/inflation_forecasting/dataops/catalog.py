from __future__ import annotations


RAW_DATASET_METADATA = {
    "name": "US state-level inflation panel",
    "path": "Data/RawData.csv",
    "grain": "One row per state-quarter observation.",
    "primary_key": ["state", "year", "quarter"],
    "time_coverage": {"start_year": 1978, "end_year": 2017, "frequency": "quarterly"},
    "notes": [
        "The repository currently contains 34 state-level entities including the District of Columbia.",
        "The panel combines non-tradable inflation, tradable inflation and headline inflation.",
    ],
}


RAW_DATA_DICTIONARY = [
    {
        "name": "state",
        "dtype": "string",
        "role": "entity_id",
        "description": "State or district name used as the panel identifier.",
        "nullable": False,
    },
    {
        "name": "year",
        "dtype": "int64",
        "role": "time_year",
        "description": "Calendar year of the quarterly observation.",
        "nullable": False,
    },
    {
        "name": "quarter",
        "dtype": "int64",
        "role": "time_quarter",
        "description": "Quarter index within the year. Valid domain: 1-4.",
        "nullable": False,
    },
    {
        "name": "pi_nt",
        "dtype": "float64",
        "role": "feature",
        "description": "Inflation rate for non-tradable goods.",
        "nullable": False,
    },
    {
        "name": "pi_t",
        "dtype": "float64",
        "role": "feature",
        "description": "Inflation rate for tradable goods.",
        "nullable": False,
    },
    {
        "name": "pi",
        "dtype": "float64",
        "role": "target",
        "description": "Headline inflation rate used as the default forecasting target.",
        "nullable": False,
    },
]


DATA_ASSETS = [
    {
        "path": "Data/RawData.csv",
        "role": "canonical_raw_dataset",
        "description": "Primary machine-readable panel used by the Python package and CLI.",
    },
    {
        "path": "Data/Data Cleaned.xlsx",
        "role": "legacy_spreadsheet",
        "description": "Legacy Excel export preserved for traceability with the original academic workflow.",
    },
    {
        "path": "Data/ARIMA Forecasting and Errors.xlsx",
        "role": "legacy_results",
        "description": "Legacy spreadsheet containing ARIMA forecasts and error calculations from the original project.",
    },
]
