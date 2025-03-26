import pandas as pd
import vectorbt as vbt
from .base import StrategyBase
from core.metrics import compute_metrics

class MultiTimeframeMomentum(StrategyBase):
    """
    Стратегія, що враховує моментум на 1m і 15m таймфреймах.
    Якщо обидва > 0 => купуємо, якщо обидва < 0 => продаємо.
    """

    def __init__(self, price_data: pd.DataFrame,
                 short_window=5,
                 long_window=5):
        """
        :param price_data: DataFrame (OHLCV)
        :param short_window: період для momentum на 1хв
        :param long_window: період для momentum на 15хв
        """
        super().__init__(price_data)
        self.short_window = short_window
        self.long_window = long_window
        self.signals = None
        self.pf = None

    def generate_signals(self):
        """
        Робимо resample('15T').last() для отримання 15-хв даних,
        потім reindex у 1-хв, обчислюємо momentums (mom_1m, mom_15m).
        Якщо mom_1m>0 і mom_15m>0 => +1, якщо <0 і <0 => -1, інакше 0.
        """
        df_wide = self._reshape_to_wide(self.price_data)
        close_1m = df_wide["close"]

        close_15m = close_1m.resample("15T").last()
        close_15m = close_15m.reindex(close_1m.index, method="ffill")

        mom_1m = (close_1m / close_1m.shift(self.short_window)) - 1.0
        mom_15m = (close_15m / close_15m.shift(self.long_window)) - 1.0

        buy_signal = ((mom_1m > 0) & (mom_15m > 0)).astype(int)
        sell_signal = ((mom_1m < 0) & (mom_15m < 0)).astype(int) * -1

        self.signals = buy_signal + sell_signal
        return self.signals

    def run_backtest(self):
        """
        Створюємо Portfolio, fees=0.1% і slippage=0.05%.
        """
        if self.signals is None:
            self.generate_signals()
        df_wide = self._reshape_to_wide(self.price_data)
        close = df_wide["close"]

        entries = self.signals == 1
        exits = self.signals == -1

        self.pf = vbt.Portfolio.from_signals(
            close,
            entries=entries,
            exits=exits,
            fees=0.001,
            slippage=0.0005,
            direction='longonly'
        )
        return self.pf

    def get_metrics(self) -> dict:
        """
        Викликає compute_metrics(self.pf).
        """
        if self.pf is None:
            raise ValueError("Run run_backtest first.")
        return compute_metrics(self.pf)

    def _reshape_to_wide(self, df_long: pd.DataFrame):
        """
        Pivot (time, symbol) -> wide
        """
        return df_long.pivot_table(
            index="time",
            columns="symbol",
            values=["open", "high", "low", "close", "volume"]
        )
