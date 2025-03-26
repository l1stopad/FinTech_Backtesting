import pandas as pd
from strategies.base import StrategyBase

class SmaCrossStrategy(StrategyBase):
    """
    Стратегія перетину двох простих ковзних середніх (SMA)
    із фільтром волатильності.
    """
    def __init__(self, price_data: pd.DataFrame, short_window: int = 10,
                 long_window: int = 50, vol_threshold: float = 0.01):
        super().__init__(price_data)
        self.short_window = short_window
        self.long_window = long_window
        self.vol_threshold = vol_threshold
        self.signals = None

    def generate_signals(self) -> pd.DataFrame:
        df_wide = self.data
        close = df_wide["close"]

        sma_short = close.rolling(self.short_window).mean()
        sma_long = close.rolling(self.long_window).mean()

        crossover = (sma_short > sma_long).astype(int) - (sma_short < sma_long).astype(int)

        # Фільтр волатильності
        daily_ret = close.pct_change()
        vol = daily_ret.rolling(1440).std()
        low_vol_mask = vol < self.vol_threshold
        crossover = crossover.where(~low_vol_mask, other=0)

        self.signals = crossover
        return self.signals

    def run_backtest(self):
        if self.signals is None:
            self.generate_signals()
        close = self.data["close"]
        entries = self.signals == 1
        exits = self.signals == -1
        return self._run_portfolio(close, entries, exits,
                                   fees=0.001, slippage=0.0005, direction='longonly')
