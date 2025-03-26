import pandas as pd
import vectorbt as vbt
from .base import StrategyBase
from core.metrics import compute_metrics

class VwapReversionStrategy(StrategyBase):
    """
    Стратегія, що робить ставку на повернення ціни до VWAP (Volume Weighted Average Price).
    Якщо ціна значно нижча за VWAP (деviation < -threshold) => купуємо,
    якщо значно вища (deviation > threshold) => продаємо.
    """

    def __init__(self, price_data: pd.DataFrame, threshold: float = 0.01):
        """
        :param price_data: DataFrame (OHLCV)
        :param threshold: відсоткове відхилення від VWAP, за яким робимо контртрендовий вхід/вихід
        """
        super().__init__(price_data)
        self.threshold = threshold
        self.signals = None
        self.pf = None

    def generate_signals(self):
        """
        Обчислює VWAP (через rolling суму price*volume / volume)
        і дивиться відхилення від фактичної ціни:
        deviation = (close - vwap) / vwap
        Якщо deviation < -threshold => +1, deviation > threshold => -1.
        """
        df_wide = self._reshape_to_wide(self.price_data)
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
        """
        Будуємо Portfolio на основі сигналів.
        """
        if self.signals is None:
            self.generate_signals()
        df_wide = self._reshape_to_wide(self.price_data)
        close = df_wide["close"]

        entries = (self.signals == 1)
        exits = (self.signals == -1)

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
        Повертає метрики з compute_metrics(self.pf).
        """
        if self.pf is None:
            raise ValueError("Run run_backtest first.")
        return compute_metrics(self.pf)

    def _reshape_to_wide(self, df_long: pd.DataFrame):
        """
        Pivot (time, symbol) -> wide формат.
        """
        return df_long.pivot_table(
            index="time",
            columns="symbol",
            values=["open", "high", "low", "close", "volume"]
        )
