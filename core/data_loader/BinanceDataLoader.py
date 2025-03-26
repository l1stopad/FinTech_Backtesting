import os
import time
import pandas as pd
from typing import List, Optional
import pyarrow as pa
import pyarrow.parquet as pq
import ccxt


class DataLoader:
    """
    Клас, що відповідає за реальне завантаження 1-хвилинних OHLCV-даних із Binance через ccxt,
    кешування у parquet та валідацію.
    """

    def __init__(
        self,
        data_path: str = "./data/btc_1m_feb25.parquet",
        start_date: str = "2025-02-01",
        end_date: str = "2025-02-28",
        symbols: Optional[List[str]] = None,
    ):
        """
        :param data_path: Шлях до локального parquet-файлу з даними.
        :param start_date: Початок періоду (YYYY-MM-DD).
        :param end_date: Кінець періоду (YYYY-MM-DD).
        :param symbols: Якщо задано, завантажимо лише ці символи. Якщо None – оберемо топ 100.
        """
        self.data_path = data_path
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.symbols = symbols if symbols else []
        self.data = None

        # ccxt-біржа
        self.binance = ccxt.binance({"enableRateLimit": True})

    def load_data(self) -> pd.DataFrame:
        """
        Основна функція для завантаження. Якщо локальний файл існує – зчитуємо.
        Якщо ні – отримуємо з Binance, кешуємо у parquet і повертаємо DataFrame.
        """
        if os.path.exists(self.data_path):
            print(f"[DataLoader] Loading data from local cache: {self.data_path}")
            self.data = pd.read_parquet(self.data_path)
        else:
            print("[DataLoader] Local data not found. Start fetching from Binance ...")
            self.data = self._fetch_and_build_dataset()

            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
            self.data.to_parquet(self.data_path, compression="snappy")
            print(f"[DataLoader] Data saved to {self.data_path}")

        # Якщо symbols задано, фільтруємо
        if self.symbols:
            self.data = self.data[self.data["symbol"].isin(self.symbols)]

        # Валідація
        self._validate_data()

        return self.data

    def get_top_liquid_symbols(self, limit: int = 100) -> List[str]:
        """
        Отримує список найбільш ліквідних пар до BTC. Використовує:
         - market-інфо від Binance (суто за 24h volume).
        """
        # Всі маркети з Binance
        markets = self.binance.fetch_markets()
        btc_markets = [m for m in markets if m["quote"] == "BTC" and m["active"]]

        # Сортуємо за об'ємом (info['quoteVolume'] або 'baseVolume'])
        # У різних полів може бути різне значення, залежить від відповіді біржі.
        # Спробуємо baseVolume.
        sorted_markets = sorted(
            btc_markets,
            key=lambda m: float(m["info"]["baseVolume"]) if ("baseVolume" in m["info"]) else 0,
            reverse=True,
        )
        top_markets = sorted_markets[:limit]

        top_symbols = [m["symbol"] for m in top_markets]
        return top_symbols

    def _fetch_and_build_dataset(self) -> pd.DataFrame:
        """
        1) Якщо self.symbols порожній – обираємо топ-100 пар.
        2) Для кожного символу отримуємо 1m OHLCV за заданий період.
        3) Об'єднуємо в один DataFrame з колонками:
           [time, symbol, open, high, low, close, volume].
        4) Повертаємо зведений DataFrame.
        """
        if not self.symbols:
            print("[DataLoader] symbols not set. Fetching top-100 liquid symbols to BTC...")
            self.symbols = self.get_top_liquid_symbols(limit=100)
            print("[DataLoader] Found top 100 symbols:", self.symbols)

        all_dfs = []
        for sym in self.symbols:
            print(f"[DataLoader] Fetching 1m data for {sym} ...")
            df_sym = self._fetch_symbol_ohlcv(sym)
            if df_sym is not None and not df_sym.empty:
                all_dfs.append(df_sym)
            else:
                print(f"[DataLoader] No data for symbol: {sym}")

        if not all_dfs:
            raise ValueError("[DataLoader] No data was fetched for any symbol.")

        df_all = pd.concat(all_dfs, ignore_index=True)
        # Сортуємо за часом
        df_all.sort_values(["symbol", "time"], inplace=True)
        return df_all

    def _fetch_symbol_ohlcv(self, symbol: str) -> pd.DataFrame:
        """
        Завантажує 1m OHLCV через ccxt.fetch_ohlcv для одного symbol
        за період [self.start_date, self.end_date].
        Повертає DataFrame зі стовпцями: [time, symbol, open, high, low, close, volume].
        """
        timeframe = "1m"
        since = int(self.start_date.timestamp() * 1000)  # у мс
        end_timestamp = int(self.end_date.timestamp() * 1000)
        limit = 1000  # Binance віддає максимум 1000-1500 свічок за запит

        ohlcv_all = []
        while True:
            # fetch_ohlcv(symbol, timeframe, since, limit)
            data = self.binance.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not data:
                break
            ohlcv_all += data

            last_ts = data[-1][0]
            # Наступний виклик - з наступної хвилини
            since = last_ts + 60_000
            if since >= end_timestamp:
                break
            # Маленька пауза, щоб не впертися в rate limit
            time.sleep(0.2)

        if not ohlcv_all:
            return pd.DataFrame()

        df = pd.DataFrame(
            ohlcv_all,
            columns=["time", "open", "high", "low", "close", "volume"]
        )
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df["symbol"] = symbol
        # Фільтруємо рядки, що виходять за межі end_date
        df = df[df["time"] <= self.end_date]
        return df

    def _validate_data(self):
        """
        Мінімальні перевірки дат та колонок.
        """
        if self.data.isnull().values.any():
            print("[DataLoader] Warning: dataset contains NaN values.")

        required = {"time", "symbol", "open", "high", "low", "close", "volume"}
        missing = required - set(self.data.columns)
        if missing:
            raise ValueError(f"[DataLoader] Missing columns: {missing}")

        # Переконуємося, що дати в рамках [start_date, end_date]
        self.data["time"] = pd.to_datetime(self.data["time"])
        mask = (self.data["time"] >= self.start_date) & (self.data["time"] <= self.end_date)
        self.data = self.data[mask]
        if self.data.empty:
            raise ValueError(
                "[DataLoader] No data in specified date range. Check start_date/end_date or the fetch logic."
            )
