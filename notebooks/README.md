# Legacy Research Assets

This directory stores the original exploratory notebooks and the Stata workflow used during the academic phase of the project.

They are not part of the production package. The supported runnable code now lives in `src/inflation_forecasting/` and the repository-level wrappers under `scripts/`.

| File | Type | Status |
|------|------|--------|
| `legacy/01_dataset_construction_and_graphs.ipynb` | Notebook | Legacy exploratory analysis. |
| `legacy/02_arima_modeling_and_forecasting.do` | Stata script | Legacy econometric workflow. |
| `legacy/03_lstm_model.ipynb` | Notebook | Legacy deep-learning experiment. |
| `legacy/04_model_selection.ipynb` | Notebook | Legacy comparison notebook. |

## Recommendation

Use the packaged CLI, API, and Streamlit app for reproducible runs, structured outputs, and test coverage. Keep these files for traceability, comparison with the original write-up, or manual exploratory work.
