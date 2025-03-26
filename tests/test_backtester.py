import pytest
import pandas as pd
import numpy as np
import os

from strategies.sma_cross import SmaCrossStrategy
from core.backtester import Backtester

@pytest.fixture
def sample_data():
    dates = pd.date_range("2025-02-01", periods=10, freq="1min")
    symbol = ["TEST/BTC"]
    idx = pd.MultiIndex.from_product([dates, symbol], names=["time", "symbol"])
    df = pd.DataFrame({
        "open": np.random.rand(10),
        "high": np.random.rand(10),
        "low": np.random.rand(10),
        "close": np.random.rand(10),
        "volume": np.random.rand(10)
    }, index=idx).reset_index()
    return df

def test_backtester_run_all(sample_data, tmp_path):
    strat = SmaCrossStrategy(sample_data)
    bt = Backtester([strat], results_path=str(tmp_path))
    bt.run_all()

    # перевірка наявності metrics.csv
    metrics_file = os.path.join(tmp_path, "metrics.csv")
    assert os.path.exists(metrics_file), "metrics.csv not found"

    # перевірка скріншотів
    eq_img = os.path.join(tmp_path, "screenshots", "SmaCrossStrategy_equity.png")
    heat_img = os.path.join(tmp_path, "screenshots", "SmaCrossStrategy_heatmap.png")
    assert os.path.exists(eq_img)
    assert os.path.exists(heat_img)
