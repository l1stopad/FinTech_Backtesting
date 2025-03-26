import pandas as pd
from strategies.base import StrategyBase

class VwapReversionStrategy(StrategyBase):
    """
    Стратегія повернення ціни до VWAP.
    Якщо deviation < -threshold => купуємо,
    якщо deviation > threshold => продаємо.
    """
    def __init__(self, price_data: pd.DataFrame, threshold: float = 0.01):
        super().__init__(price_data)
        self.threshold = threshold
        self.signals = None

    def generate_signals(self) -> pd.DataFrame:
        df_wide = self.data  # вже перетворено в wide-формат
        close = df_wide["close"]
        volume = df_wide["volume"]

        price_times_vol = close * volume
        rolling_pv = price_times_vol.rolling(1440).sum()
        rolling_vol = volume.rolling(1440).sum()
        vwap = rolling_pv / rolling_vol

        deviation = (close - vwap) / vwap
        buy_signal = (deviation < -self.threshold).astype(int)
        sell_signal = (deviation > self.threshold).astype(int) * -1

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
