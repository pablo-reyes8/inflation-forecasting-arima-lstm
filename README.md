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

| File / Notebook | Purpose |
|-----------------|---------|
| **`Dataset_Construction_and_Graphs.ipynb`** | Loads the raw CPI data, builds a tidy time-series DataFrame, and produces exploratory plots (rolling mean/variance, additive decomposition, box-plots by period). |
| **`ARIMA_Modeling_and_Forecasting.do`** | End-to-end ARIMA script in Stata: stationarity tests, ACF/PACF diagnostics, model selection via AIC/BIC, residual checks, dynamic forecasts. |
| **`LSTM_Model.ipynb`** | Prepares supervised windows, tunes a multi-layer LSTM with Keras, trains the configuration, and evaluates out-of-sample accuracy (R², MSE, MAE). |
| **`Model_Selection.ipynb`** | Brings ARIMA and LSTM forecasts together, visualises residuals and forecast paths, computes final metrics, and selects the most robust model for deployment. |

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

Execute the notebooks in the order listed above (1 → 4) to reproduce every
figure, metric and forecast.

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests at
https://github.com/pablo-reyes8

---

## License

Released under the MIT License – free for personal or commercial use.

