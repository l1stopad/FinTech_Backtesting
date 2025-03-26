from abc import ABC, abstractmethod
import pandas as pd
import vectorbt as vbt
from core.metrics import compute_metrics

class StrategyBase(ABC):
    """
    Абстрактний базовий клас для торгової стратегії.
    Він містить спільні методи для перетворення даних та створення портфеля.
    """
    def __init__(self, price_data: pd.DataFrame):
        """
        :param price_data: DataFrame із колонками [time, symbol, open, high, low, close, volume]
        """
        self.price_data = price_data
        self.raw_data = price_data
        self.data = self._reshape_to_wide(price_data)
        self.pf = None

    @abstractmethod
    def generate_signals(self) -> pd.DataFrame:
        """
        Має згенерувати сигнали (1 = вхід, -1 = вихід/шорт, 0 = тримати).
        """
        pass

    @abstractmethod
    def run_backtest(self):
        """
        Запускає бектест на основі generate_signals.
        """
        pass

    def get_metrics(self) -> dict:
        """
        Повертає метрики портфеля через compute_metrics.
        """
        if self.pf is None:
            raise ValueError("Спочатку запустіть run_backtest.")
        return compute_metrics(self.pf)

    def _reshape_to_wide(self, df_long: pd.DataFrame) -> pd.DataFrame:
        """
        Перетворює дані з long-формату у wide (pivot по time та symbol).
        """
        return df_long.pivot_table(
            index="time",
            columns="symbol",
            values=["open", "high", "low", "close", "volume"]
        )

    def _run_portfolio(self, close: pd.DataFrame, entries: pd.DataFrame, exits: pd.DataFrame,
                       fees: float = 0.001, slippage: float = 0.0005, direction: str = 'longonly'):
        """
        Створює портфель на основі сигналів із заданими параметрами.
        """
        self.pf = vbt.Portfolio.from_signals(
            close,
            entries=entries,
            exits=exits,
            fees=fees,
            slippage=slippage,
            direction=direction
        )
        return self.pf
