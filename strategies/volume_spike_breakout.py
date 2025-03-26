import pandas as pd
import vectorbt as vbt
from .base import StrategyBase
from core.metrics import compute_metrics

class VolumeSpikeBreakout(StrategyBase):
    """
    Стратегія, що шукає спайк обсягу (volume) + пробиття локального max,
    і входить у лонг. Вихід, коли ціна падає нижче локального мінімуму.
    """

    def __init__(self, price_data: pd.DataFrame,
                 lookback: int = 20,
                 volume_mult: float = 2.0):
        """
        :param price_data: DataFrame (OHLCV)
        :param lookback: Скільки свічок для визначення середнього volume і локального max/min
        :param volume_mult: Множник std, щоб вважати volume "спайком"
        """
        super().__init__(price_data)
        self.lookback = lookback
        self.volume_mult = volume_mult
        self.signals = None
        self.pf = None

    def generate_signals(self):
        """
        Визначаємо spike = volume > mean(volume)+volume_mult*std(volume) (по lookback).
        Якщо spike і ціна > rolling_high => buy=1,
        якщо ціна < rolling_low => sell=-1.
        """
        df_wide = self._reshape_to_wide(self.price_data)
        close = df_wide["close"]
        volume = df_wide["volume"]

        vol_mean = volume.rolling(self.lookback).mean()
        vol_std = volume.rolling(self.lookback).std()
        spike = volume > (vol_mean + self.volume_mult * vol_std)

        rolling_high = close.rolling(self.lookback).max()
        rolling_low = close.rolling(self.lookback).min()

        buy_signal = (spike & (close > rolling_high)).astype(int)
        sell_signal = (close < rolling_low).astype(int) * -1

        signals = buy_signal + sell_signal
        self.signals = signals
        return signals

    def run_backtest(self):
        """
        Формуємо Portfolio.
        """
        if self.signals is None:
            self.generate_signals()

        df_wide = self._reshape_to_wide(self.price_data)
        close = df_wide["close"]
        entries = (self.signals == 1)
        exits = (self.signals == -1)

        self.pf = vbt.Portfolio.from_signals(
            close,
            entries,
            exits,
            fees=0.001,
            slippage=0.0005,
            direction='longonly'
        )
        return self.pf

    def get_metrics(self) -> dict:
        """
        Повертає метрики (у т.ч. win_rate) через compute_metrics.
        """
        if self.pf is None:
            raise ValueError("Run run_backtest first.")
        return compute_metrics(self.pf)

    def _reshape_to_wide(self, df_long: pd.DataFrame):
        """
        Перетворює long => wide формат.
        """
        return df_long.pivot_table(
            index="time",
            columns="symbol",
            values=["open", "high", "low", "close", "volume"]
        )
