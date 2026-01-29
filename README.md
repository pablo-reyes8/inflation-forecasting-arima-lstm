# Inflation Forecasting: Econometrics + Machine Learning

![Repo size](https://img.shields.io/github/repo-size/pablo-reyes8/inflation-forecasting-arima-lstm)
![Last commit](https://img.shields.io/github/last-commit/pablo-reyes8/inflation-forecasting-arima-lstm)
![Open issues](https://img.shields.io/github/issues/pablo-reyes8/inflation-forecasting-arima-lstm)
![Contributors](https://img.shields.io/github/contributors/pablo-reyes8/inflation-forecasting-arima-lstm)
![Forks](https://img.shields.io/github/forks/pablo-reyes8/inflation-forecasting-arima-lstm?style=social)
![Stars](https://img.shields.io/github/stars/pablo-reyes8/inflation-forecasting-arima-lstm?style=social)

Production-style inflation-forecasting pipeline that combines classical ARIMA diagnostics in Stata with modern ML/DL models in Python. It covers data preparation, exploratory analysis, model selection, residual checks, dynamic multi-step forecasts, and a clean benchmarking suite with MSE, MAE, RMSE and R^2.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Highlights](#highlights)
- [Repository Structure](#repository-structure)
- [Models Included](#models-included)
- [Quickstart](#quickstart)
- [CLI Usage](#cli-usage)
- [Docker](#docker)
- [Results and Reporting](#results-and-reporting)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

This repository formalizes a full inflation-forecasting workflow:

1. **Stata econometrics** for diagnostics and classical model selection (ADF, ACF/PACF, ARIMA candidates).
2. **Python modeling** for LSTM/GRU tuning, baselines, and reproducible evaluation.
3. **CLI and artifacts** for repeatable training, inference, and metrics export.

The goal is to provide a professional, reproducible project structure that is easy to extend and present.

---

## Highlights

- **Dual-stack approach**: Econometric rigor + ML/DL performance.
- **Multi-model benchmarks**: ARIMA/ARMA/SARIMA/ARIMAX, ARCH/GARCH, LSTM/GRU, Prophet, and ML baselines.
- **Dynamic forecasts**: Multi-step out-of-sample predictions for policy or investment use cases.
- **Reproducible outputs**: All metrics and predictions are saved to `outputs/`.
- **Clean modular code**: Reusable Python package with unit tests.

---

## Repository Structure

| Path | Purpose |
|------|---------|
| `Scripts/` | Original notebooks and Stata scripts (kept for reference). |
| `src/inflation_forecasting/` | Python package (data, modeling, evaluation, CLI). |
| `src/inflation_forecasting/models/econometria/` | Econometric models (ARIMA/ARMA/SARIMA/ARIMAX/ARCH/GARCH). |
| `src/inflation_forecasting/models/ml/` | ML/DL models (baselines, LSTM, GRU). |
| `Data/` | Raw and processed datasets (CSV/XLSX). |
| `tests/` | Pytest unit tests for core utilities. |
| `outputs/` | Auto-generated metrics and prediction artifacts. |
| `Dockerfile` | Container for reproducible runs. |
| `requirements.txt` | Python dependencies. |

---

## Models Included

**Econometrics**
- ARIMA, ARMA, SARIMA, ARIMAX
- ARCH, GARCH
- HP filter and seasonal decomposition

**Machine Learning / Deep Learning**
- LSTM, GRU (with hyperparameter tuning)
- Random Forest, Gradient Boosting, Linear Regression
- XGBoost
- Prophet

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## CLI Usage

```bash
# Descriptive stats and decomposition
inflation-forecast describe --state Maryland
inflation-forecast hp-filter --state Maryland --lamb 1600
inflation-forecast decompose --state Maryland --period 4

# ARIMA (fixed order or grid search)
inflation-forecast arima --order 1,0,3
inflation-forecast arima --grid --p-min 0 --p-max 3 --d-min 0 --d-max 1 --q-min 0 --q-max 3

# Econometric extensions
inflation-forecast arma --order 1,1
inflation-forecast sarima --order 1,0,1 --auto-seasonal
inflation-forecast arimax --order 1,0,1 --exog-cols pi_nt,pi_t
inflation-forecast arch --p 1
inflation-forecast garch --p 1 --q 1

# LSTM / GRU
inflation-forecast lstm-train --look-back 4 --epochs 80 --forecast-steps 4 --save-model
inflation-forecast lstm-tune --look-back 4 --max-trials 10
inflation-forecast gru-train --look-back 4 --epochs 80

# LSTM inference from saved artifacts
inflation-forecast lstm-forecast --model-path outputs/lstm_model_*.keras --scaler-path outputs/lstm_scaler_*.joblib --steps 4

# Prophet and ML baselines
inflation-forecast prophet
inflation-forecast ml-train --model random_forest --lags 4
inflation-forecast ml-train --model xgboost --lags 4
```

---

## Docker

```bash
docker build -t inflation-forecast .
docker run --rm -v "$PWD:/app" inflation-forecast describe --state Maryland
```

---

## Results and Reporting

All runs export metrics and predictions into `outputs/` as CSV/JSON. This makes it easy to build reports, dashboards or client-ready visualizations.

---

## Dependencies

**Optional dependencies by model**
- ARIMA/HP filter/decomposition: `statsmodels`
- ARCH/GARCH: `arch`
- LSTM/GRU + tuning: `tensorflow`, `keras-tuner`
- Prophet: `prophet`
- XGBoost: `xgboost`

---

## Contributing

Contributions are welcome. Please open an issue or submit a pull request at:
https://github.com/pablo-reyes8

---

## License

Released under the MIT License - free for personal or commercial use.
