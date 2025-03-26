import pandas as pd
import ta
import vectorbt as vbt
from .base import StrategyBase
from core.metrics import compute_metrics

class RsiBbStrategy(StrategyBase):
    """
    Стратегія, що використовує RSI і Bollinger Bands для входу/виходу.
    Якщо RSI < 30 і ціна пробиває нижню межу BB знизу вгору => вхід у лонг.
    Якщо RSI > 70 => вихід.
    """

    def __init__(self, price_data: pd.DataFrame,
                 rsi_window: int = 14,
                 bb_window: int = 20,
                 bb_std: float = 2.0):
        """
        :param price_data: DataFrame із колонками [time, symbol, open, high, low, close, volume]
        :param rsi_window: вікно RSI
        :param bb_window: вікно Bollinger Bands
        :param bb_std: множник std для Bollinger (звичайно 2.0)
        """
        super().__init__(price_data)
        self.rsi_window = rsi_window
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.signals = None
        self.pf = None

    def generate_signals(self) -> pd.DataFrame:
        """
        Обчислює RSI і нижню межу Bollinger.
        Якщо RSI < 30 і ціна перетнула нижню BB знизу вверх => +1,
        Якщо RSI > 70 => -1,
        інакше 0.
        """
        df_wide = self._reshape_to_wide(self.price_data)
        close = df_wide["close"]

        rsi = close.apply(
            lambda col: ta.momentum.RSIIndicator(col, window=self.rsi_window).rsi()
        )

        def bb_apply(col):
            bb = ta.volatility.BollingerBands(col, window=self.bb_window, window_dev=self.bb_std)
            return pd.DataFrame({
                "lower": bb.bollinger_lband(),
                "upper": bb.bollinger_hband()
            })

        bb_dict = {}
        for c in close.columns:
            bb_dict[c] = bb_apply(close[c])

        signals_list = []
        for c in close.columns:
            c_rsi = rsi[c]
            c_lower = bb_dict[c]["lower"]
            c_close = close[c]

            # Умови входу/виходу
            buy_signal = ((c_rsi < 30) &
                          (c_close > c_lower) &
                          (c_close.shift(1) <= c_lower.shift(1))).astype(int)
            sell_signal = (c_rsi > 70).astype(int) * -1
            c_signals = buy_signal + sell_signal
            c_signals.name = c
            signals_list.append(c_signals)

        signals_df = pd.concat(signals_list, axis=1)
        self.signals = signals_df
        return signals_df

    def run_backtest(self):
        """
        Формує портфель із entries=1, exits=-1, комісія і сліпейдж.
        :return: vectorbt.Portfolio
        """
        if self.signals is None:
            self.generate_signals()

        df_wide = self._reshape_to_wide(self.price_data)
        close = df_wide["close"]
        entries = (self.signals == 1)
        exits = (self.signals == -1)

        self.pf = vbt.Portfolio.from_signals(
            close, entries, exits,
            fees=0.00075,
            slippage=0.0005,
            direction='longonly'
        )
        return self.pf

    def get_metrics(self) -> dict:
        """
        Викликає compute_metrics(self.pf).

        :return: dict
        """
        if self.pf is None:
            raise ValueError("Run run_backtest first.")
        return compute_metrics(self.pf)

    def _reshape_to_wide(self, df_long: pd.DataFrame):
        """
        Pivot таблиця з (time, symbol) у багаторівневий формат.
        """
        return df_long.pivot_table(
            index="time",
            columns="symbol",
            values=["open", "high", "low", "close", "volume"]
        )
