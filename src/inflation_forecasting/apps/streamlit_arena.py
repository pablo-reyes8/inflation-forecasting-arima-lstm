from __future__ import annotations

from dataclasses import asdict
import html
import json
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ..datasets.io import load_raw_data
from .arena import (
    MODEL_CATALOG,
    ArenaDataset,
    ArenaRunConfig,
    available_model_catalog,
    build_leaderboard_frame,
    build_predictions_frame,
    parse_order_text,
    prepare_arena_dataset,
    read_tabular_data,
    run_model_arena,
)


DEFAULT_MODELS = ["arima", "sarima", "prophet", "linear_regression", "random_forest"]


def _parse_order_or_stop(label: str, text: str, expected: int) -> tuple[int, ...]:
    try:
        return parse_order_text(text, expected)
    except ValueError as exc:
        st.sidebar.error(f"{label}: {exc}")
        st.stop()


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(230, 244, 241, 0.95), transparent 35%),
                radial-gradient(circle at top right, rgba(255, 239, 214, 0.9), transparent 30%),
                linear-gradient(180deg, #f5f7f2 0%, #f9faf7 100%);
        }
        .hero-card {
            padding: 1.5rem 1.6rem;
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(244, 249, 247, 0.90));
            border: 1px solid rgba(18, 66, 60, 0.12);
            box-shadow: 0 14px 36px rgba(21, 55, 52, 0.10);
            margin-bottom: 1.1rem;
        }
        .metric-card {
            padding: 1rem 1.2rem;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(18, 66, 60, 0.10);
            min-height: 118px;
        }
        .metric-label {
            color: #4a5e58;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .metric-value {
            color: #102f2c;
            font-size: 1.65rem;
            font-weight: 700;
            margin-top: 0.4rem;
        }
        .metric-subtle {
            color: #5e726c;
            font-size: 0.9rem;
            margin-top: 0.35rem;
        }
        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.85rem;
        }
        .chip {
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            padding: 0.4rem 0.7rem;
            border-radius: 999px;
            background: rgba(22, 74, 68, 0.08);
            border: 1px solid rgba(22, 74, 68, 0.10);
            color: #184944;
            font-size: 0.84rem;
            font-weight: 600;
        }
        .chip-accent {
            background: rgba(204, 122, 0, 0.10);
            border-color: rgba(204, 122, 0, 0.12);
            color: #8f5600;
        }
        .panel-card {
            padding: 1rem 1.1rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(18, 66, 60, 0.10);
            box-shadow: 0 10px 26px rgba(21, 55, 52, 0.06);
            margin-bottom: 1rem;
        }
        .panel-kicker {
            color: #5b6f69;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }
        .panel-title {
            color: #102f2c;
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .panel-copy {
            color: #4f6661;
            font-size: 0.95rem;
            line-height: 1.5;
        }
        .step-card {
            padding: 0.95rem 1rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(18, 66, 60, 0.09);
            min-height: 162px;
        }
        .step-number {
            color: #8f5600;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-weight: 700;
        }
        .step-title {
            color: #102f2c;
            font-size: 1rem;
            font-weight: 700;
            margin: 0.3rem 0 0.45rem;
        }
        .step-copy {
            color: #58706a;
            font-size: 0.92rem;
            line-height: 1.45;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _metric_card(label: str, value: str, subtle: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-subtle">{subtle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _chip_html(values: list[str], *, accent: bool = False) -> str:
    css_class = "chip chip-accent" if accent else "chip"
    return "".join(f'<span class="{css_class}">{html.escape(value)}</span>' for value in values)


def _panel_card(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="panel-card">
          <div class="panel-kicker">{html.escape(kicker)}</div>
          <div class="panel-title">{html.escape(title)}</div>
          <div class="panel-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _step_card(number: int, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="step-card">
          <div class="step-number">Step {number}</div>
          <div class="step-title">{html.escape(title)}</div>
          <div class="step-copy">{html.escape(copy)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _dataset_profile_frame(dataset: ArenaDataset) -> pd.DataFrame:
    rows = []
    for column in dataset.frame.columns:
        series = dataset.frame[column]
        sample = series.dropna().iloc[0] if series.notna().any() else None
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "missing": int(series.isna().sum()),
                "sample": None if sample is None else str(sample)[:40],
            }
        )
    return pd.DataFrame(rows)


def _model_catalog_frame(catalog: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for key, spec in catalog.items():
        rows.append(
            {
                "model": spec["label"],
                "family": spec["family"],
                "available": "yes" if spec["available"] else "no",
                "dependencies": ", ".join(spec["requires"]),
                "notes": spec.get("reason") or "Ready in current setup.",
            }
        )
    return pd.DataFrame(rows)


def _guide_view(dataset: ArenaDataset, selected_models: list[str], config: ArenaRunConfig) -> None:
    catalog = available_model_catalog(has_exog=bool(dataset.exog_cols))
    st.markdown("### Workflow")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _step_card(1, "Load data", "Use the built-in state panel or upload a CSV/XLSX file with a date column or year-quarter fields.")
    with c2:
        _step_card(2, "Define the arena", "Pick the models, shared split, ranking metric, and optional econometric or sequence-model settings.")
    with c3:
        _step_card(3, "Run the benchmark", "All selected models are evaluated on the same validation and test windows so comparisons stay honest.")
    with c4:
        _step_card(4, "Inspect the winner", "Review leaderboard scores, prediction traces, and download CSV/JSON artifacts for downstream work.")

    left, right = st.columns([1.15, 1.0])
    with left:
        _panel_card(
            "Upload Contract",
            "Accepted dataset layouts",
            "Use either a single date column or a year/quarter pair. The target and any optional exogenous regressors must be numeric.",
        )
        st.code(
            "date,value,exog_1\n2020-03-31,1.42,0.33\n2020-06-30,1.55,0.29\n2020-09-30,1.60,0.31",
            language="csv",
        )
        _panel_card(
            "Current Scenario",
            dataset.name,
            (
                f"Target: <strong>{html.escape(dataset.target_col)}</strong><br>"
                f"Observations: <strong>{dataset.metadata['rows']}</strong><br>"
                f"Split: <strong>{config.train_size:.0%}/{config.val_size:.0%}/{config.test_size:.0%}</strong>"
            ),
        )
    with right:
        st.markdown("### Model Catalog")
        st.dataframe(_model_catalog_frame(catalog), use_container_width=True, hide_index=True)
        if selected_models:
            st.markdown("### Current Selection")
            st.markdown(
                f'<div class="chip-row">{_chip_html([MODEL_CATALOG[key]["label"] for key in selected_models], accent=True)}</div>',
                unsafe_allow_html=True,
            )


@st.cache_data(show_spinner=False)
def _load_builtin_frame() -> pd.DataFrame:
    return load_raw_data()


def _dataset_panel() -> ArenaDataset:
    st.sidebar.markdown("## Data")
    st.sidebar.caption("Use the built-in panel or upload your own quarterly series to compare multiple forecasting families.")
    source = st.sidebar.radio("Source", options=["Repository panel", "Upload your own series"], index=0)

    if source == "Repository panel":
        raw = _load_builtin_frame()
        st.sidebar.caption("Source dataset: `Data/RawData.csv`")
        state = st.sidebar.selectbox("State", sorted(raw["state"].unique().tolist()), index=0)
        target = st.sidebar.selectbox("Target", ["pi", "pi_nt", "pi_t"], index=0)
        exog_choices = [column for column in ["pi_nt", "pi_t", "pi"] if column != target]
        exog_cols = st.sidebar.multiselect("Optional exogenous columns", exog_choices, default=[])
        return prepare_arena_dataset(
            raw,
            dataset_name=f"Repository / {state}",
            target_col=target,
            year_col="year",
            quarter_col="quarter",
            entity_col="state",
            entity_value=state,
            exog_cols=exog_cols,
            interpolate_missing=True,
        )

    uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    if uploaded is None:
        st.info("Upload a dataset to activate the arena.")
        st.stop()

    raw = read_tabular_data(uploaded.name, uploaded.getvalue())
    st.sidebar.caption(f"Loaded `{uploaded.name}` with {len(raw)} rows and {len(raw.columns)} columns.")
    columns = raw.columns.tolist()
    has_calendar_cols = "year" in columns and "quarter" in columns
    date_mode = st.sidebar.radio(
        "Time index",
        options=["Date column", "Year + quarter"] if has_calendar_cols else ["Date column"],
        index=0,
    )
    date_col = st.sidebar.selectbox("Date column", columns) if date_mode == "Date column" else None
    year_col = st.sidebar.selectbox("Year column", columns, index=columns.index("year")) if date_mode == "Year + quarter" else None
    quarter_col = (
        st.sidebar.selectbox("Quarter column", columns, index=columns.index("quarter"))
        if date_mode == "Year + quarter"
        else None
    )
    candidate_numeric = raw.select_dtypes(include=["number"]).columns.tolist() or columns
    target_col = st.sidebar.selectbox("Target column", candidate_numeric)
    entity_col = st.sidebar.selectbox("Entity column", ["<none>", *columns], index=0)
    entity_col = None if entity_col == "<none>" else entity_col
    entity_value = None
    if entity_col:
        values = raw[entity_col].dropna().astype(str).sort_values().unique().tolist()
        entity_value = st.sidebar.selectbox("Entity value", values)
    exog_options = [column for column in candidate_numeric if column != target_col]
    exog_cols = st.sidebar.multiselect("Optional exogenous columns", exog_options, default=[])
    interpolate_missing = st.sidebar.checkbox("Interpolate missing values", value=True)
    return prepare_arena_dataset(
        raw,
        dataset_name=f"Upload / {uploaded.name}",
        target_col=target_col,
        date_col=date_col,
        year_col=year_col,
        quarter_col=quarter_col,
        entity_col=entity_col,
        entity_value=entity_value,
        exog_cols=exog_cols,
        interpolate_missing=interpolate_missing,
    )


def _config_panel(dataset: ArenaDataset) -> tuple[list[str], ArenaRunConfig]:
    st.sidebar.markdown("## Arena")
    st.sidebar.caption("Benchmark every selected model on the same split and rank them with a single validation metric.")
    catalog = available_model_catalog(has_exog=bool(dataset.exog_cols))
    available_options = [key for key, spec in catalog.items() if spec["available"]]
    default_models = [model for model in DEFAULT_MODELS if model in available_options] or available_options[:3]
    selected_models = st.sidebar.multiselect(
        "Models to compare",
        options=list(MODEL_CATALOG.keys()),
        default=default_models,
        format_func=lambda key: MODEL_CATALOG[key]["label"],
    )

    unavailable = {key: spec for key, spec in catalog.items() if not spec["available"]}
    if unavailable:
        with st.sidebar.expander("Unavailable models"):
            for key, spec in unavailable.items():
                st.caption(f"{spec['label']}: {spec['reason']}")

    train_size = st.sidebar.slider("Train share", min_value=0.5, max_value=0.8, value=0.6, step=0.05)
    val_size = st.sidebar.slider("Validation share", min_value=0.1, max_value=0.25, value=0.2, step=0.05)
    test_size = round(1.0 - train_size - val_size, 2)
    if test_size <= 0:
        st.sidebar.error("Train + validation must be below 1.0.")
        st.stop()
    st.sidebar.caption(f"Test share: {test_size:.2f}")
    ranking_metric = st.sidebar.selectbox("Leaderboard metric", ["rmse", "mae", "mape", "smape", "r2"], index=0)

    with st.sidebar.expander("Econometric settings"):
        arima_order = _parse_order_or_stop("ARIMA order", st.text_input("ARIMA order (p,d,q)", value="1,0,1"), 3)
        arma_order = _parse_order_or_stop("ARMA order", st.text_input("ARMA order (p,q)", value="1,1"), 2)
        sarima_order = _parse_order_or_stop("SARIMA order", st.text_input("SARIMA order (p,d,q)", value="1,0,1"), 3)
        sarima_seasonal_text = st.text_input("SARIMA seasonal order (P,D,Q,s)", value="")
        sarima_seasonal_order = (
            _parse_order_or_stop("SARIMA seasonal order", sarima_seasonal_text, 4)
            if sarima_seasonal_text.strip()
            else None
        )

    with st.sidebar.expander("ML / DL settings"):
        lags = st.slider("Lag features", min_value=2, max_value=12, value=4)
        look_back = st.slider("Sequence look_back", min_value=2, max_value=12, value=4)
        epochs = st.slider("DL epochs", min_value=10, max_value=200, value=40, step=10)
        batch_size = st.select_slider("DL batch size", options=[8, 16, 32, 64], value=16)
        validation_split = st.slider("Internal validation split", min_value=0.0, max_value=0.3, value=0.1, step=0.05)
        patience = st.slider("Early stopping patience", min_value=3, max_value=20, value=8)
        random_seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
        lstm_units = st.slider("LSTM units", min_value=16, max_value=256, value=64, step=16)
        gru_units = st.slider("GRU units", min_value=16, max_value=256, value=64, step=16)
        dense_units = st.slider("Dense units", min_value=16, max_value=128, value=32, step=16)
        dropout = st.slider("Dropout", min_value=0.0, max_value=0.5, value=0.1, step=0.05)
        learning_rate = st.select_slider("Learning rate", options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2], value=1e-3)

    config = ArenaRunConfig(
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        ranking_metric=ranking_metric,
        arima_order=arima_order,
        arma_order=arma_order,
        sarima_order=sarima_order,
        sarima_seasonal_order=sarima_seasonal_order,
        lags=lags,
        look_back=look_back,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        patience=patience,
        random_seed=int(random_seed),
        lstm_units=lstm_units,
        lstm_dense_units=dense_units,
        gru_units=gru_units,
        gru_dense_units=dense_units,
        dropout=dropout,
        learning_rate=learning_rate,
    )
    return selected_models, config


def _overview(dataset: ArenaDataset, selected_models: list[str], config: ArenaRunConfig) -> None:
    metadata = dataset.metadata
    selected_labels = [MODEL_CATALOG[key]["label"] for key in selected_models] or ["No models selected"]
    exog_labels = list(dataset.exog_cols) or ["No exogenous regressors"]
    st.markdown(
        f"""
        <div class="hero-card">
          <h1 style="margin:0;color:#0f2e2b;">Inflation Forecast Arena</h1>
          <p style="margin:0.5rem 0 0;color:#4f6661;max-width:900px;">
            Compare econometric, machine-learning and deep-learning models on a shared train / validation / test split.
            Use the repository panel or upload your own quarterly series, then inspect a single leaderboard, detailed prediction traces, and exportable artifacts.
          </p>
          <div class="chip-row">{_chip_html(selected_labels, accent=True)}</div>
          <div class="chip-row">{_chip_html(exog_labels)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _metric_card("Dataset", dataset.name, f"{metadata['rows']} observations")
    with col2:
        _metric_card("Coverage", f"{metadata['start'].date()} to {metadata['end'].date()}", f"Frequency: {metadata['frequency'] or 'irregular'}")
    with col3:
        _metric_card("Models", str(len(selected_models)), ", ".join(MODEL_CATALOG[key]["label"] for key in selected_models[:3]) or "None")
    with col4:
        _metric_card("Split", f"{config.train_size:.0%} / {config.val_size:.0%} / {config.test_size:.0%}", f"{len(dataset.exog_cols)} exogenous columns")

    left, right = st.columns([1.1, 1.0])
    with left:
        _panel_card(
            "Scenario",
            f"{dataset.target_col} for {dataset.entity_value or dataset.name}",
            (
                f"Coverage from <strong>{metadata['start'].date()}</strong> to <strong>{metadata['end'].date()}</strong>. "
                f"The arena ranks models by <strong>{config.ranking_metric.upper()}</strong> on the validation window before you inspect test behavior."
            ),
        )
    with right:
        _panel_card(
            "Run Recipe",
            "What happens when you click Run arena",
            (
                f"Selected models: <strong>{len(selected_models)}</strong><br>"
                f"Validation metric: <strong>{config.ranking_metric.upper()}</strong><br>"
                f"Sequence look_back: <strong>{config.look_back}</strong> | Lag features: <strong>{config.lags}</strong>"
            ),
        )


def _split_chart(dataset: ArenaDataset, config: ArenaRunConfig) -> go.Figure:
    series = dataset.series
    train_end = int(len(series) * config.train_size)
    val_end = train_end + int(len(series) * config.val_size)
    split_labels = pd.Series("test", index=series.index)
    split_labels.iloc[:train_end] = "train"
    split_labels.iloc[train_end:val_end] = "validation"
    chart_df = pd.DataFrame({"date": series.index, "value": series.values, "split": split_labels.values})
    fig = px.line(chart_df, x="date", y="value", color="split", color_discrete_map={"train": "#1f5c57", "validation": "#cc7a00", "test": "#7d8b84"})
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), legend_title_text="")
    return fig


def _leaderboard_view(leaderboard: pd.DataFrame, ranking_metric: str) -> None:
    successful = leaderboard.loc[leaderboard["status"] == "ok"].copy()
    if successful.empty:
        st.warning("No successful runs yet. Check dependency warnings or model errors.")
        st.dataframe(leaderboard, use_container_width=True)
        return

    sort_column = f"validation_{ranking_metric}"
    if sort_column in successful.columns:
        ascending = ranking_metric != "r2"
        successful = successful.sort_values(sort_column, ascending=ascending)

    col1, col2, col3 = st.columns(3)
    best = successful.iloc[0]
    with col1:
        _metric_card("Best validation model", best["model"], f"{sort_column}: {best.get(sort_column, float('nan')):.4f}")
    with col2:
        _metric_card("Best test RMSE", f"{successful['test_rmse'].min():.4f}", "Across successful models")
    with col3:
        _metric_card("Successful runs", str(len(successful)), f"Errors/skips: {(leaderboard['status'] != 'ok').sum()}")

    bar_source = successful[["model", sort_column]].rename(columns={sort_column: "score"})
    bar_fig = px.bar(bar_source, x="model", y="score", color="model", height=340)
    bar_fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=20, b=10))

    scatter_source = successful.copy()
    test_column = f"test_{ranking_metric}"
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(bar_fig, use_container_width=True)
    with chart_col2:
        if test_column in scatter_source.columns:
            scatter_fig = px.scatter(
                scatter_source,
                x=sort_column,
                y=test_column,
                color="family",
                size="duration_seconds",
                hover_name="model",
                text="model",
                height=340,
            )
            scatter_fig.update_traces(textposition="top center")
            scatter_fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), legend_title_text="")
            st.plotly_chart(scatter_fig, use_container_width=True)
        else:
            st.info("Need both validation and test metrics to render the comparison map.")

    st.dataframe(successful, use_container_width=True)


def _predictions_chart(predictions: pd.DataFrame, model_key: str) -> go.Figure:
    selected = predictions.loc[predictions["model_key"] == model_key].reset_index()
    fig = go.Figure()
    for phase, dash in {"validation": "dot", "test": "solid"}.items():
        phase_df = selected.loc[selected["phase"] == phase]
        if phase_df.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=phase_df["date"],
                y=phase_df["actual"],
                mode="lines",
                name=f"Actual ({phase})",
                line=dict(color="#1f5c57", dash=dash),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=phase_df["date"],
                y=phase_df["predicted"],
                mode="lines",
                name=f"Predicted ({phase})",
                line=dict(color="#cc7a00", dash=dash),
            )
        )
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), legend_title_text="")
    return fig


def _detail_view(results: list[Any], predictions: pd.DataFrame) -> None:
    successful = [result for result in results if result.status == "ok"]
    if not successful:
        st.info("Run the arena to inspect model-level detail.")
        return
    selected_key = st.selectbox(
        "Model detail",
        options=[result.model_key for result in successful],
        format_func=lambda key: MODEL_CATALOG[key]["label"],
    )
    result = next(result for result in successful if result.model_key == selected_key)
    top1, top2, top3 = st.columns(3)
    with top1:
        _metric_card("Model", result.label, result.family)
    with top2:
        _metric_card("Validation RMSE", f"{result.validation_metrics.get('rmse', float('nan')):.4f}", "Shared validation window")
    with top3:
        _metric_card("Test RMSE", f"{result.test_metrics.get('rmse', float('nan')):.4f}", f"Duration: {result.duration_seconds or 0:.2f}s")

    trace_tab, metrics_tab, config_tab, history_tab = st.tabs(["Trace", "Metrics", "Configuration", "Training history"])
    with trace_tab:
        st.plotly_chart(_predictions_chart(predictions, selected_key), use_container_width=True)
        trace_df = predictions.loc[predictions["model_key"] == selected_key].reset_index()
        st.dataframe(trace_df, use_container_width=True, hide_index=True)
    with metrics_tab:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Validation metrics")
            st.json(result.validation_metrics)
        with col2:
            st.markdown("### Test metrics")
            st.json(result.test_metrics)
    with config_tab:
        if result.notes:
            st.info("\n".join(result.notes))
        st.markdown("### Parameters")
        st.code(json.dumps(result.parameters, indent=2, default=str), language="json")
    with history_tab:
        if result.history is not None and not result.history.empty:
            history_fig = px.line(result.history, x="epoch", y="loss", color="phase", height=320)
            st.plotly_chart(history_fig, use_container_width=True)
            st.dataframe(result.history, use_container_width=True)
        else:
            st.info("This model did not produce a training-history artifact.")


def _downloads_view(
    leaderboard: pd.DataFrame,
    predictions: pd.DataFrame,
    results: list[Any],
    context: dict[str, Any] | None,
) -> None:
    st.markdown("### Export run artifacts")
    st.caption("Use the CSV files for spreadsheets or quick plotting, and the JSON bundle for downstream services or notebooks.")

    summary_payload = {
        "context": context,
        "leaderboard": leaderboard.to_dict(orient="records"),
        "results": [
            {
                "model_key": result.model_key,
                "label": result.label,
                "family": result.family,
                "status": result.status,
                "validation_metrics": result.validation_metrics,
                "test_metrics": result.test_metrics,
                "parameters": result.parameters,
                "duration_seconds": result.duration_seconds,
                "error": result.error,
            }
            for result in results
        ],
    }

    st.download_button(
        label="Download leaderboard CSV",
        data=leaderboard.to_csv(index=False).encode("utf-8"),
        file_name="arena_leaderboard.csv",
        mime="text/csv",
    )
    st.download_button(
        label="Download predictions CSV",
        data=predictions.to_csv().encode("utf-8"),
        file_name="arena_predictions.csv",
        mime="text/csv",
    )
    st.download_button(
        label="Download arena summary JSON",
        data=json.dumps(summary_payload, indent=2, default=str).encode("utf-8"),
        file_name="arena_summary.json",
        mime="application/json",
    )


def main() -> None:
    st.set_page_config(
        page_title="Inflation Forecast Arena",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()

    dataset = _dataset_panel()
    selected_models, config = _config_panel(dataset)
    _overview(dataset, selected_models, config)

    guide_tab, data_tab, leaderboard_tab, detail_tab, downloads_tab = st.tabs(
        ["Guide", "Data", "Arena", "Model detail", "Downloads"]
    )
    with guide_tab:
        _guide_view(dataset, selected_models, config)
    with data_tab:
        left, right = st.columns([1.35, 1.0])
        with left:
            st.plotly_chart(_split_chart(dataset, config), use_container_width=True)
            st.markdown("### Prepared series")
            st.dataframe(dataset.frame.head(20), use_container_width=True)
        with right:
            _panel_card(
                "Dataset Profile",
                dataset.name,
                (
                    f"Target column: <strong>{dataset.target_col}</strong><br>"
                    f"Rows after preparation: <strong>{dataset.metadata['rows']}</strong><br>"
                    f"Frequency: <strong>{dataset.metadata['frequency'] or 'irregular'}</strong>"
                ),
            )
            st.markdown("### Column profile")
            st.dataframe(_dataset_profile_frame(dataset), use_container_width=True, hide_index=True)
            st.markdown("### Upload quick reference")
            st.code(
                "date,value,exog_1\n2020-03-31,1.42,0.33\n2020-06-30,1.55,0.29",
                language="csv",
            )

    run_button = st.sidebar.button("Run arena", type="primary", use_container_width=True)
    if run_button:
        if not selected_models:
            st.sidebar.error("Select at least one model.")
            st.stop()
        progress = st.sidebar.progress(0)
        status = st.sidebar.empty()
        status.write("Running model arena...")
        results = run_model_arena(dataset, selected_models, config)
        st.session_state["arena_results"] = results
        st.session_state["arena_leaderboard"] = build_leaderboard_frame(results)
        st.session_state["arena_predictions"] = build_predictions_frame(results)
        st.session_state["arena_context"] = {
            "dataset": {
                "name": dataset.name,
                "target": dataset.target_col,
                "entity": dataset.entity_value,
                "exogenous_columns": list(dataset.exog_cols),
                "metadata": dataset.metadata,
            },
            "selected_models": selected_models,
            "config": asdict(config),
        }
        progress.progress(100)
        status.write("Arena finished.")

    leaderboard = st.session_state.get("arena_leaderboard")
    predictions = st.session_state.get("arena_predictions")
    results = st.session_state.get("arena_results")
    context = st.session_state.get("arena_context")

    with leaderboard_tab:
        if leaderboard is None:
            st.info("Configure the data and models, then run the arena.")
        else:
            _leaderboard_view(leaderboard, config.ranking_metric)
            if (leaderboard["status"] != "ok").any():
                with st.expander("Failed or skipped models"):
                    st.dataframe(leaderboard.loc[leaderboard["status"] != "ok"], use_container_width=True)

    with detail_tab:
        if results is None or predictions is None:
            st.info("Run the arena to inspect model predictions.")
        else:
            _detail_view(results, predictions)

    with downloads_tab:
        if leaderboard is None or predictions is None or results is None:
            st.info("Downloads become available after the first arena run.")
        else:
            _downloads_view(leaderboard, predictions, results, context)


if __name__ == "__main__":
    main()
