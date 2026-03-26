import pandas as pd

from inflation_forecasting.models.ml.common import create_supervised, infer_future_index


def test_create_supervised_builds_expected_windows():
    X, y = create_supervised(pd.Series([1, 2, 3, 4, 5]).to_numpy(), look_back=2)
    assert X.shape == (3, 2)
    assert y.shape == (3,)


def test_infer_future_index_keeps_quarterly_spacing():
    series = pd.Series([1.0, 1.2, 1.3, 1.5], index=pd.date_range("2020-03-31", periods=4, freq="QE-DEC"))
    future_index = infer_future_index(series, steps=2)
    assert len(future_index) == 2
    assert future_index[0] > series.index[-1]
