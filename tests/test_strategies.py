import pytest
import pandas as pd
import numpy as np


from strategies.sma_cross import SmaCrossStrategy
from strategies.rsi_bb import RsiBbStrategy
from strategies.vwap_reversion import VwapReversionStrategy
from strategies.multi_tf_momentum import MultiTimeframeMomentum
from strategies.atr_trailing_breakout import AtrTrailingBreakout
from strategies.volume_spike_breakout import VolumeSpikeBreakout

@pytest.fixture
def sample_data():
    dates = pd.date_range("2025-02-01", periods=60, freq="1min")
    symbols = ["ETH/BTC", "BNB/BTC"]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["time", "symbol"])
    df = pd.DataFrame({
        "open": np.random.rand(120)*100,
        "high": np.random.rand(120)*100,
        "low": np.random.rand(120)*100,
        "close": np.random.rand(120)*100,
        "volume": np.random.rand(120)*10,
    }, index=idx).reset_index()
    return df

def test_sma_cross_signals(sample_data):
    strat = SmaCrossStrategy(sample_data, short_window=5, long_window=10)
    signals = strat.generate_signals()
    assert signals.shape[0] == 60
    assert signals.shape[1] == 2

def test_sma_cross_backtest(sample_data):
    strat = SmaCrossStrategy(sample_data)
    pf = strat.run_backtest()
    metrics = strat.get_metrics()
    for k in ["total_return", "sharpe_ratio", "max_drawdown", "win_rate"]:
        assert k in metrics

def test_rsi_bb_signals(sample_data):
    strat = RsiBbStrategy(sample_data, rsi_window=14, bb_window=20, bb_std=2.0)
    signals = strat.generate_signals()
    assert signals.shape[0] == 60
    assert signals.shape[1] == 2

def test_rsi_bb_backtest(sample_data):
    strat = RsiBbStrategy(sample_data)
    pf = strat.run_backtest()
    metrics = strat.get_metrics()
    assert "total_return" in metrics

def test_vwap_reversion_signals(sample_data):
    strat = VwapReversionStrategy(sample_data, threshold=0.01)
    signals = strat.generate_signals()
    assert signals.shape[0] == 60
    assert signals.shape[1] == 2

def test_vwap_reversion_backtest(sample_data):
    strat = VwapReversionStrategy(sample_data)
    pf = strat.run_backtest()
    metrics = strat.get_metrics()
    assert "sharpe_ratio" in metrics

def test_multi_tf_signals(sample_data):
    strat = MultiTimeframeMomentum(sample_data, short_window=5, long_window=3)
    signals = strat.generate_signals()
    assert signals.shape == (60, 2)

def test_multi_tf_backtest(sample_data):
    strat = MultiTimeframeMomentum(sample_data)
    pf = strat.run_backtest()
    metrics = strat.get_metrics()
    assert "win_rate" in metrics

def test_atr_trailing_breakout_signals(sample_data):
    strat = AtrTrailingBreakout(sample_data, lookback=10, atr_period=5, atr_mult=2.0)
    signals = strat.generate_signals()
    assert signals.shape == (60, 2)

def test_atr_trailing_breakout_backtest(sample_data):
    strat = AtrTrailingBreakout(sample_data)
    pf = strat.run_backtest()
    metrics = strat.get_metrics()
    assert "total_return" in metrics

def test_volume_spike_breakout_signals(sample_data):
    strat = VolumeSpikeBreakout(sample_data, lookback=10, volume_mult=2.0)
    signals = strat.generate_signals()
    assert signals.shape == (60, 2)

def test_volume_spike_breakout_backtest(sample_data):
    strat = VolumeSpikeBreakout(sample_data)
    pf = strat.run_backtest()
    metrics = strat.get_metrics()
    assert "sharpe_ratio" in metrics
