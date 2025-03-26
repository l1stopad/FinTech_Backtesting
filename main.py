import os
from core.data_loader.BinanceDataLoader import DataLoader
from core.backtester import Backtester


from strategies.sma_cross import SmaCrossStrategy
from strategies.rsi_bb import RsiBbStrategy
from strategies.vwap_reversion import VwapReversionStrategy
from strategies.multi_tf_momentum import MultiTimeframeMomentum
from strategies.atr_trailing_breakout import AtrTrailingBreakout
from strategies.volume_spike_breakout import VolumeSpikeBreakout

def main():
    # 1. Завантаження даних
    loader = DataLoader(
        data_path="./data/btc_1m_feb25.parquet",
        start_date="2025-02-01",
        end_date="2025-02-28",
        symbols=None  # якщо None, підхопить топ-100 ліквідних пар
    )
    data = loader.load_data()

    # 2. Створюємо екземпляри стратегій
    sma_strategy = SmaCrossStrategy(data)
    rsi_strategy = RsiBbStrategy(data)
    vwap_strategy = VwapReversionStrategy(data)
    multi_tf_strat = MultiTimeframeMomentum(data)
    atr_strat = AtrTrailingBreakout(data)
    volume_spike_strat = VolumeSpikeBreakout(data)

    # 3. Запускаємо бектест
    bt = Backtester(
        strategies=[
            sma_strategy,
            rsi_strategy,
            vwap_strategy,
            multi_tf_strat,
            atr_strat,
            volume_spike_strat
        ],
        results_path="./results"
    )
    bt.run_all()

if __name__ == "__main__":
    main()
