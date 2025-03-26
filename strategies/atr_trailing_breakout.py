import pandas as pd
import ta
from strategies.base import StrategyBase

class AtrTrailingBreakout(StrategyBase):
    """
    Стратегія: вхід при пробитті локального максимуму,
    вихід, якщо ціна падає нижче (rolling_high - ATR * atr_mult).
    """
    def __init__(self, price_data: pd.DataFrame, lookback: int = 20,
                 atr_period: int = 14, atr_mult: float = 2.0):
        super().__init__(price_data)
        self.lookback = lookback
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.signals = None

    def generate_signals(self) -> pd.DataFrame:
        df_wide = self.data
        close = df_wide["close"]
        high = df_wide["high"]
        low = df_wide["low"]

        def atr_apply(h, l, c):
            atr_ind = ta.volatility.AverageTrueRange(h, l, c, window=self.atr_period)
            return atr_ind.average_true_range()

        # Обчислюємо ATR для кожного символу
        atr_list = [atr_apply(high[col], low[col], close[col]).rename(col) for col in close.columns]
        atr_df = pd.concat(atr_list, axis=1)

        rolling_high = close.rolling(self.lookback).max()
        buy_signal = (close > rolling_high).astype(int)
        exit_signal = (close < (rolling_high - self.atr_mult * atr_df)).astype(int) * -1

        self.signals = buy_signal + exit_signal
        return self.signals

    def run_backtest(self):
        if self.signals is None:
            self.generate_signals()
        close = self.data["close"]
        entries = self.signals == 1
        exits = self.signals == -1
        return self._run_portfolio(close, entries, exits,
                                   fees=0.001, slippage=0.0005, direction='longonly')
