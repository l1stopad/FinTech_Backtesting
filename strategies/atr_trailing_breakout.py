import pandas as pd
import vectorbt as vbt
import ta
from .base import StrategyBase
from core.metrics import compute_metrics

class AtrTrailingBreakout(StrategyBase):
    """
    Стратегія: вхід при пробитті локального макс (rolling_high),
    вихід, якщо ціна падає нижче (rolling_high - ATR * atr_mult).
    """

    def __init__(self, price_data: pd.DataFrame,
                 lookback: int = 20,
                 atr_period: int = 14,
                 atr_mult: float = 2.0):
        """
        :param price_data: DataFrame (OHLCV)
        :param lookback: скільки свічок дивитися, щоб знайти локальний максимум
        :param atr_period: період ATR
        :param atr_mult: множник ATR, який віднімаємо від локального максимуму
        """
        super().__init__(price_data)
        self.lookback = lookback
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.signals = None
        self.pf = None

    def generate_signals(self):
        """
        Для кожного символу рахуємо ATR, rolling_high (lookback).
        Якщо close > rolling_high => buy=1,
        Якщо close < (rolling_high - atr*atr_mult) => sell=-1.
        """
        df_wide = self._reshape_to_wide(self.price_data)
        close = df_wide["close"]
        high = df_wide["high"]
        low = df_wide["low"]

        def atr_apply(h, l, c):
            atr_ind = ta.volatility.AverageTrueRange(h, l, c, window=self.atr_period)
            return atr_ind.average_true_range()

        atr_list = []
        for col in close.columns:
            atr_col = atr_apply(high[col], low[col], close[col])
            atr_list.append(atr_col.rename(col))
        atr_df = pd.concat(atr_list, axis=1)

        rolling_high = close.rolling(self.lookback).max()
        buy_signal = (close > rolling_high).astype(int)
        exit_signal = (close < (rolling_high - self.atr_mult * atr_df)).astype(int) * -1

        signals = buy_signal + exit_signal
        self.signals = signals
        return signals

    def run_backtest(self):
        """
        Створює Portfolio зі звичайними комісіями/сліпейджем.
        """
        if self.signals is None:
            self.generate_signals()
        df_wide = self._reshape_to_wide(self.price_data)
        close = df_wide["close"]

        entries = self.signals == 1
        exits = self.signals == -1

        pf = vbt.Portfolio.from_signals(
            close,
            entries=entries,
            exits=exits,
            fees=0.001,
            slippage=0.0005,
            direction='longonly'
        )
        self.pf = pf
        return pf

    def get_metrics(self) -> dict:
        """
        Використовує compute_metrics(self.pf).
        """
        if self.pf is None:
            raise ValueError("Run run_backtest first.")
        return compute_metrics(self.pf)

    def _reshape_to_wide(self, df_long: pd.DataFrame):
        """
        Pivot (time, symbol).
        """
        return df_long.pivot_table(
            index="time",
            columns="symbol",
            values=["open", "high", "low", "close", "volume"]
        )
