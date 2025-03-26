import pandas as pd
from strategies.base import StrategyBase

class VolumeSpikeBreakout(StrategyBase):
    """
    Стратегія пробою локального максимуму з врахуванням спайку обсягу.
    """
    def __init__(self, price_data: pd.DataFrame, lookback: int = 20, volume_mult: float = 2.0):
        super().__init__(price_data)
        self.lookback = lookback
        self.volume_mult = volume_mult
        self.signals = None

    def generate_signals(self) -> pd.DataFrame:
        df_wide = self.data
        close = df_wide["close"]
        volume = df_wide["volume"]

        vol_mean = volume.rolling(self.lookback).mean()
        vol_std = volume.rolling(self.lookback).std()
        spike = volume > (vol_mean + self.volume_mult * vol_std)

        rolling_high = close.rolling(self.lookback).max()
        rolling_low = close.rolling(self.lookback).min()

        buy_signal = (spike & (close > rolling_high)).astype(int)
        sell_signal = (close < rolling_low).astype(int) * -1

        self.signals = buy_signal + sell_signal
        return self.signals

    def run_backtest(self):
        if self.signals is None:
            self.generate_signals()
        close = self.data["close"]
        entries = self.signals == 1
        exits = self.signals == -1
        return self._run_portfolio(close, entries, exits,
                                   fees=0.001, slippage=0.0005, direction='longonly')
