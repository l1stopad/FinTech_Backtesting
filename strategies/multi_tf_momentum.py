import pandas as pd
from strategies.base import StrategyBase

class MultiTimeframeMomentum(StrategyBase):
    """
    Стратегія, що враховує моментум на 1-хв та 15-хв таймфреймах.
    Вхід, якщо обидва > 0, вихід, якщо обидва < 0.
    """
    def __init__(self, price_data: pd.DataFrame, short_window: int = 5, long_window: int = 5):
        super().__init__(price_data)
        self.short_window = short_window
        self.long_window = long_window
        self.signals = None

    def generate_signals(self) -> pd.DataFrame:
        df_wide = self.data
        close_1m = df_wide["close"]

        # Отримуємо 15-хв дані через resample і ffill
        close_15m = close_1m.resample("15T").last().reindex(close_1m.index, method="ffill")

        mom_1m = (close_1m / close_1m.shift(self.short_window)) - 1.0
        mom_15m = (close_15m / close_15m.shift(self.long_window)) - 1.0

        buy_signal = ((mom_1m > 0) & (mom_15m > 0)).astype(int)
        sell_signal = ((mom_1m < 0) & (mom_15m < 0)).astype(int) * -1

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
