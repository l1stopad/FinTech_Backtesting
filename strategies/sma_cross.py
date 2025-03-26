import pandas as pd
import vectorbt as vbt
from .base import StrategyBase
from core.metrics import compute_metrics

class SmaCrossStrategy(StrategyBase):
    """
    Стратегія перетину двох простих ковзних середніх (SMA).
    Додатково містить фільтр волатильності, за якого не торгуємо, якщо
    середньодобова std < vol_threshold.
    """

    def __init__(self, price_data: pd.DataFrame,
                 short_window: int = 10,
                 long_window: int = 50,
                 vol_threshold: float = 0.01):
        """
        :param price_data: DataFrame із колонками [time, symbol, open, high, low, close, volume]
        :param short_window: період короткого SMA
        :param long_window: період довгого SMA
        :param vol_threshold: поріг волатильності, нижче якого сигнали "відключаємо"
        """
        super().__init__(price_data)
        self.short_window = short_window
        self.long_window = long_window
        self.vol_threshold = vol_threshold
        self.signals = None
        self.pf = None

    def generate_signals(self) -> pd.DataFrame:
        """
        Обчислює коротке та довге SMA. Якщо коротке > довге => сигнал = +1,
        коротке < довге => сигнал = -1, інакше 0.
        Якщо std(daily_ret, 1 day) < vol_threshold => сигнал = 0 (не торгуємо).
        :return: DataFrame signals (індекс = час, колонки = символи).
        """
        df_wide = self._reshape_to_wide(self.price_data)
        close = df_wide["close"]

        sma_short = close.rolling(self.short_window).mean()
        sma_long = close.rolling(self.long_window).mean()

        # buy=1, sell=-1
        crossover = (sma_short > sma_long).astype(int) - (sma_short < sma_long).astype(int)

        # Фільтр волатильності
        daily_ret = close.pct_change()
        vol = daily_ret.rolling(1440).std()
        low_vol_mask = vol < self.vol_threshold
        # Зануляємо де волатильність низька
        crossover = crossover.where(~low_vol_mask, other=0)

        self.signals = crossover
        return self.signals

    def run_backtest(self):
        """
        Використовує сигнали generate_signals() і створює Portfolio
        із комісією 0.1% та невеликим slippage. Лонгова торгівля (longonly).

        :return: vectorbt.Portfolio
        """
        if self.signals is None:
            self.generate_signals()

        df_wide = self._reshape_to_wide(self.price_data)
        close = df_wide["close"]
        entries = self.signals == 1
        exits = self.signals == -1

        self.pf = vbt.Portfolio.from_signals(
            close, entries, exits,
            fees=0.001,      # 0.1%
            slippage=0.0005, # 0.05%
            direction='longonly'
        )
        return self.pf

    def get_metrics(self) -> dict:
        """
        Повертає ключові метрики, обчислені через compute_metrics.

        :return: dict
        """
        if self.pf is None:
            raise ValueError("Please run run_backtest first.")
        return compute_metrics(self.pf)

    def _reshape_to_wide(self, df_long: pd.DataFrame):
        """
        Перетворює вихідний DataFrame з колонки "symbol" та "time" у "wide" формат
        з індексом=час і колонками (open, high, low, close, volume).
        :param df_long: DataFrame (long format)
        :return: DataFrame (wide format) із багаторівневими колонками
        """
        return df_long.pivot_table(
            index="time",
            columns="symbol",
            values=["open", "high", "low", "close", "volume"]
        )
