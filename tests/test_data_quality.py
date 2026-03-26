import pandas as pd
import pytest

from inflation_forecasting.dataops.quality import assert_quality_gate, audit_inflation_dataset


def test_audit_inflation_dataset_detects_issues():
    df = pd.DataFrame(
        {
            "state": ["A", "A", "A"],
            "year": [2020, 2020, 2020],
            "quarter": [1, 1, 5],
            "pi_nt": [1.0, 1.1, None],
            "pi_t": [0.5, 0.6, 0.7],
            "pi": [0.8, 0.9, 1.0],
        }
    )
    report = audit_inflation_dataset(df)
    assert report.duplicates_on_primary_key == 1
    assert report.invalid_quarters == 1
    assert report.missing_by_column["pi_nt"] == 1
    assert len(report.issues) == 3


def test_assert_quality_gate_raises_on_report_issues():
    df = pd.DataFrame(
        {
            "state": ["A", "A"],
            "year": [2020, 2020],
            "quarter": [1, 1],
            "pi_nt": [1.0, 1.1],
            "pi_t": [0.5, 0.6],
            "pi": [0.8, 0.9],
        }
    )
    report = audit_inflation_dataset(df)
    with pytest.raises(ValueError):
        assert_quality_gate(report)
