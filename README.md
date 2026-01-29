# Inflation Forecasting with ARIMA and LSTM


![Repo size](https://img.shields.io/github/repo-size/pablo-reyes8/inflation-forecasting-arima-lstm)
![Last commit](https://img.shields.io/github/last-commit/pablo-reyes8/inflation-forecasting-arima-lstm)
![Open issues](https://img.shields.io/github/issues/pablo-reyes8/inflation-forecasting-arima-lstm)
![Contributors](https://img.shields.io/github/contributors/pablo-reyes8/inflation-forecasting-arima-lstm)
![Forks](https://img.shields.io/github/forks/pablo-reyes8/inflation-forecasting-arima-lstm?style=social)
![Stars](https://img.shields.io/github/stars/pablo-reyes8/inflation-forecasting-arima-lstm?style=social)


A full-stack inflation-forecasting toolkit that pairs classical ARIMA diagnostics in Stata with an LSTM pipeline in Python. The project walks from raw CPI data ingestion and exploratory visualisation through model selection, hyper-parameter tuning, residual stress-testing, and one-year dynamic forecasts—culminating in a side-by-side benchmark of statistical and deep-learning approaches, complete with R², MSE, MAE, and plots for powerful insights.

---

## Repository Contents

| File / Folder | Purpose |
|-----------------|---------|
| **`Scripts/`** | Original notebooks and Stata script (kept for reference). |
| **`src/inflation_forecasting/`** | Python package with reusable data, modeling, and evaluation modules. |
| **`src/inflation_forecasting/models/econometria/`** | Modelos econometricos (ARIMA/ARMA/SARIMA/ARIMAX/ARCH/GARCH). |
| **`src/inflation_forecasting/models/ml/`** | Modelos ML y deep learning (baselines, LSTM, GRU). |
| **`Data/`** | Raw and processed datasets (CSV/XLSX). |
| **`tests/`** | Pytest unit tests for core utilities. |
| **`outputs/`** | Auto-generated metrics and prediction artifacts from CLI runs. |
| **`Dockerfile`** | Containerized environment for reproducible runs. |
| **`requirements.txt`** | Python dependencies. |

---
## Key Highlights

* **Dual-approach comparison** – side-by-side performance of statistical (ARIMA) and neural (LSTM) models.  
* **Rigorous residual diagnostics** – inverse-root, Ljung–Box, white noise tests visualised in one place.
* **Two-tier significance analysis** – every stationarity test, parameter t-stat, and residual check is evaluated at both 5 % and 1 % confidence levels to ensure the models remain robust under stricter criteria.
* **Dynamic forecasts** – one-year ahead projections with confidence bands for each candidate model.  
* **Metric suite** – in-sample \(R^2\) plus out-of-sample MSE, MAE and RMSE ensure fair benchmarking.  
* **Modular notebooks** – run independently or as a pipeline; easy to swap datasets or horizons.  
---

## Key Findings

- **ARIMA(1, 0, 3) emerges as the in-sample champion**  
  – Highest training \(R^2\) (0.868) and the most tightly centred residuals.  

- **LSTM generalises best out-of-sample**  
  – Validation \(R^2\) = 0.639, RMSE = 0.94, MAE = 0.79 – the only model that maintains strong accuracy when confronted with unseen data.  

- **Two-tier significance testing matters**  
  – Employing both 5 % and 1 % thresholds revealed that the original series was only weakly stationary at 5 %; differencing at 1 % produced alternative ARIMA candidates (4,1,1) and (2,1,3).  

- **Residual diagnostics confirm white noise**  
  – Inverse-root plots, Ljung–Box Q, and cumulative periodograms show no remaining autocorrelation or unit roots in the chosen models.  

- **Forecast paths converge**  
  – All models project a mild uptick in inflation over the next four quarters, clustering around 1.5 %–2 %.  

- **Practical takeaway**  
  – Classical ARIMA provides transparent, interpretable baselines, but the LSTM offers a superior balance of fit and generalisation, making it the recommended choice for forward-looking policy or investment scenarios.


## How to Run

You can still run the notebooks in `Scripts/`, but the project now ships with a
CLI so you can train and evaluate models from the terminal.

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### CLI examples

```bash
# Descriptive stats and decomposition
inflation-forecast describe --state Maryland
inflation-forecast hp-filter --state Maryland --lamb 1600
inflation-forecast decompose --state Maryland --period 4

# ARIMA (fixed order or grid search)
inflation-forecast arima --order 1,0,3
inflation-forecast arima --grid --p-min 0 --p-max 3 --d-min 0 --d-max 1 --q-min 0 --q-max 3

# Econometria adicional
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

**Optional dependencies by model**
- ARIMA/HP filter/decomposition: `statsmodels`
- ARCH/GARCH: `arch`
- LSTM/GRU + tuning: `tensorflow`, `keras-tuner`
- Prophet: `prophet`
- XGBoost: `xgboost`

### Docker

```bash
docker build -t inflation-forecast .
docker run --rm -v \"$PWD:/app\" inflation-forecast describe --state Maryland
```

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests at
https://github.com/pablo-reyes8

---

## License

Released under the MIT License – free for personal or commercial use.

