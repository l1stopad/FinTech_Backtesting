from abc import ABC, abstractmethod
import pandas as pd

class StrategyBase(ABC):
    """
    Абстрактний базовий клас для торгової стратегії
    """

    def __init__(self, price_data: pd.DataFrame):
        """
        :param price_data: DataFrame, що містить колонки:
             [time, symbol, open, high, low, close, volume]
        """
        self.price_data = price_data

    @abstractmethod
    def generate_signals(self) -> pd.DataFrame:
        """
        Має згенерувати сигнали (1 = вхід лонг, -1 = вихід/шорт, 0 = тримати)
        """
        pass

    @abstractmethod
    def run_backtest(self):
        """
        Запускає бектест на основі self.generate_signals() та повертає vectorbt.Portfolio
        """
        pass

    @abstractmethod
    def get_metrics(self) -> dict:
        """
        Повертає словник з ключовими метриками: total_return, sharpe_ratio, max_drawdown, тощо
        """
        pass
