import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ohlcv():
    rng = np.random.default_rng(7)
    n = 260
    rets = rng.normal(0.0004, 0.011, n)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = rng.integers(1_000_000, 5_000_000, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )
