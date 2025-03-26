import pandas as pd
import ta
from strategies.base import StrategyBase

class RsiBbStrategy(StrategyBase):
    """
    Стратегія на базі RSI та Bollinger Bands.
    Якщо RSI < 30 та ціна пробиває нижню межу BB знизу вгору – вхід,
    якщо RSI > 70 – вихід.
    """
    def __init__(self, price_data: pd.DataFrame, rsi_window: int = 14,
                 bb_window: int = 20, bb_std: float = 2.0):
        super().__init__(price_data)
        self.rsi_window = rsi_window
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.signals = None

    def generate_signals(self) -> pd.DataFrame:
        df_wide = self.data
        close = df_wide["close"]

        rsi = close.apply(lambda col: ta.momentum.RSIIndicator(col, window=self.rsi_window).rsi())

        def bb_apply(col):
            bb = ta.volatility.BollingerBands(col, window=self.bb_window, window_dev=self.bb_std)
            return pd.DataFrame({
                "lower": bb.bollinger_lband(),
                "upper": bb.bollinger_hband()
            })

        bb_dict = {c: bb_apply(close[c]) for c in close.columns}

        signals_list = []
        for c in close.columns:
            c_rsi = rsi[c]
            c_lower = bb_dict[c]["lower"]
            c_close = close[c]

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
        if self.signals is None:
            self.generate_signals()
        close = self.data["close"]
        entries = self.signals == 1
        exits = self.signals == -1
        return self._run_portfolio(close, entries, exits,
                                   fees=0.00075, slippage=0.0005, direction='longonly')
