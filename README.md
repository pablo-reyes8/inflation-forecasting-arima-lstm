# Inflation Forecasting: Research-Grade ARIMA + Deep Learning Pipeline

![Repo size](https://img.shields.io/github/repo-size/pablo-reyes8/inflation-forecasting-arima-lstm)
![Last commit](https://img.shields.io/github/last-commit/pablo-reyes8/inflation-forecasting-arima-lstm)
![Open issues](https://img.shields.io/github/issues/pablo-reyes8/inflation-forecasting-arima-lstm)
![Contributors](https://img.shields.io/github/contributors/pablo-reyes8/inflation-forecasting-arima-lstm)
![Forks](https://img.shields.io/github/forks/pablo-reyes8/inflation-forecasting-arima-lstm?style=social)
![Stars](https://img.shields.io/github/stars/pablo-reyes8/inflation-forecasting-arima-lstm?style=social)

This repository compares classical autoregressive models and modern sequence models for state-level inflation forecasting. It now ships as a reproducible Python package with modular CLIs, dataset contracts, structured run artifacts, and tests intended for serious research workflows rather than a one-off university submission.

## What Changed

- `src/` is now split into dedicated packages for `clis`, `data`, `dataops`, `artifacts`, and `models`.
- The CLI saves each run into a structured `outputs/runs/<timestamp>_<command>/` directory.
- Every training or audit run writes a machine-readable `manifest.yml`.
- Dataset quality checks were added through `inflation-forecast data-audit`.
- LSTM/GRU training was hardened with train-only scaling, deterministic seeds, time-aware validation, and training histories.
- Data assets and legacy scripts are now documented in [`Data/README.md`](Data/README.md) and [`Scripts/README.md`](Scripts/README.md).

## Repository Layout

| Path | Purpose |
|------|---------|
| `src/inflation_forecasting/clis/` | Modular command-line interface grouped by domain. |
| `src/inflation_forecasting/data/` | Data loading, preprocessing, features, and time-based splitting. |
| `src/inflation_forecasting/dataops/` | Dataset contract metadata and quality controls. |
| `src/inflation_forecasting/artifacts/` | Run directory creation and YAML manifest generation. |
| `src/inflation_forecasting/models/econometria/` | ARIMA, ARMA, SARIMA, ARIMAX, ARCH/GARCH, Prophet. |
| `src/inflation_forecasting/models/ml/` | LSTM, GRU, and tabular ML baselines. |
| `Data/` | Canonical data plus data dictionary and storage notes. |
| `Scripts/` | Legacy notebooks and Stata scripts kept for traceability. |
| `tests/` | Unit tests for preprocessing, metrics, artifacts, quality checks and splits. |
| `outputs/` | Structured run artifacts and forecasts. |

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

## Core Workflows

```bash
# Dataset quality gate
inflation-forecast data-audit --strict

# Descriptive analysis
inflation-forecast describe --state Maryland
inflation-forecast hp-filter --state Maryland --lamb 1600
inflation-forecast decompose --state Maryland --period 4

# Econometric models
inflation-forecast arima --order 1,0,3 --forecast-steps 4
inflation-forecast arima --grid --p-min 0 --p-max 3 --d-min 0 --d-max 1 --q-min 0 --q-max 3
inflation-forecast arma --order 1,1
inflation-forecast sarima --order 1,0,1 --auto-seasonal
inflation-forecast arimax --order 1,0,1 --exog-cols pi_nt,pi_t
inflation-forecast arch --p 1
inflation-forecast garch --p 1 --q 1
inflation-forecast prophet --forecast-steps 4

# Deep learning
inflation-forecast lstm-train --look-back 4 --epochs 80 --forecast-steps 4 --save-model
inflation-forecast lstm-tune --look-back 4 --max-trials 10
inflation-forecast gru-train --look-back 4 --epochs 80
inflation-forecast lstm-forecast --model-path outputs/runs/<run_id>/model.keras --scaler-path outputs/runs/<run_id>/scaler.joblib

# Tabular baselines
inflation-forecast ml-train --model random_forest --lags 4
inflation-forecast ml-train --model xgboost --lags 4
```

## Streamlit Arena

The repository now includes an interactive Streamlit app for side-by-side model comparison on either the built-in panel or your own uploaded series.

```bash
pip install -e ".[app]"
streamlit run streamlit_app.py
```

You can also launch it through the packaged entrypoint:

```bash
inflation-forecast-arena
```

The app supports:

- Uploading CSV/XLSX series
- Date or year-quarter indexing
- Optional exogenous regressors
- Shared train / validation / test splits
- Leaderboards across ARIMA, SARIMA, Prophet, ML baselines, LSTM and GRU when dependencies are available
- Downloadable comparison tables and prediction traces

## Data Contract

The canonical dataset is `Data/RawData.csv`.

- Shape: `4699 x 6`
- Grain: one row per `state-year-quarter`
- Coverage: `1978` to `2017`
- Entities: `34` state-level series including the District of Columbia
- Default target: `pi`

Field-level definitions live in [`Data/data_dictionary.yml`](Data/data_dictionary.yml).

## Artifact Contract

Each CLI run writes a self-contained directory under `outputs/runs/`, for example:

```text
outputs/runs/20260326T190001Z_lstm-train/
├── history.csv
├── manifest.yml
├── metrics.json
├── model.keras
├── predictions.csv
└── scaler.joblib
```

The manifest records:

- Command and parameterization
- Dataset slice and target metadata
- Metrics
- Relative artifact paths
- Python/runtime metadata

Artifact conventions are documented in [`outputs/README.md`](outputs/README.md).

## Legacy Scripts

The original Stata script and notebooks are kept in `Scripts/` as historical reference. The recommended production workflow is the Python package + CLI because it is reproducible, testable, and artifact-aware.

## Testing

```bash
python -m pytest -q
```

## Docker

```bash
docker build -t inflation-forecast .
docker run --rm -v "$PWD:/app" inflation-forecast data-audit
```

## License

Released under the MIT License.
