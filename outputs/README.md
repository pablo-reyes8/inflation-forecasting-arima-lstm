# Output Artifacts

The CLI writes reproducible, self-contained run directories under `outputs/runs/`.

## Standard Structure

```text
outputs/runs/<timestamp>_<command>/
├── manifest.yml
├── metrics.json
├── predictions.csv
└── ...
```

Additional files depend on the command:

- `forecast.csv` for future forecasts
- `history.csv` for neural-network training curves
- `leaderboard.csv` for grid searches
- `model.keras` and `scaler.joblib` for persisted LSTM artifacts
- `quality_report.json` and `quality_report.yml` for DataOps checks

## Manifest Fields

Each `manifest.yml` documents:

- Run identifier and UTC timestamp
- Command and parameter values
- Dataset slice metadata
- Model family and model name
- Evaluation metrics
- Relative paths to the generated artifacts
- Runtime metadata

This structure is intended to support replication, peer review, and downstream automation.
