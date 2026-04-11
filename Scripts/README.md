# Repository Scripts

This directory contains runnable repository-level wrappers and launchers.

These files are useful when working directly from the repo checkout. The installable package entrypoints are still defined in `pyproject.toml`.

## Files

| File | Purpose |
|------|---------|
| `forecast.py` | Launch the packaged CLI from a source checkout. |
| `arena.py` | Launch the Streamlit arena from a source checkout. |
| `api.py` | Launch the FastAPI service from a source checkout. |

## Recommendation

For installed usage, prefer:

- `inflation-forecast`
- `inflation-forecast-arena`
- `inflation-forecast-api`
