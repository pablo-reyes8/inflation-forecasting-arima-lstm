# Legacy Scripts

These files are retained as historical reference for the original project workflow.

They are not part of the production package. The supported code paths now live under `src/inflation_forecasting/` through the CLI, API, and Streamlit app.

| File | Type | Status |
|------|------|--------|
| `1. DataSet Constuction and Graphs.ipynb` | Notebook | Legacy exploratory analysis. |
| `2. ARIMA Modeling and Forecasting.do` | Stata script | Legacy econometric workflow. |
| `3. LTSM model.ipynb` | Notebook | Legacy deep-learning experiment. |
| `4. Model Selection.ipynb` | Notebook | Legacy comparison notebook. |

## Recommendation

Use the Python package and CLI for reproducible runs, structured outputs, and test coverage. Keep these scripts for auditability, comparison with the original write-up, or manual exploratory work.
