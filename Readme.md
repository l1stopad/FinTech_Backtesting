# 📈 Finance - Python Backtesting Framework

A modular, scalable backtesting framework for testing trading strategies on 1-minute Binance OHLCV data using **VectorBT**.

---

## 🌟 Features

- Modular architecture
- Multiple strategy support
- Backtest on top 100 BTC pairs
- Slippage and fee simulation
- Data caching and validation
- Output: metrics, charts, interactive HTML reports
- Unit-tested core logic

---

## ⚙️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/l1stopad/FinTech_Backtesting
```

2. **Create and activate virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate      # Windows
# or
source venv/bin/activate   # Linux/macOS
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install Kaleido (for saving charts as PNG):**
```bash
pip install -U kaleido
```

---

## 📊 Usage

1. **Download data and run all strategies:**
```bash
python main.py
```

> ✅ Data will be saved to `./data/btc_1m_feb25.parquet`  
> ✅ Results will be saved to `./results/`

---

## 📅 Data(you can change)

- 100 most liquid BTC pairs on Binance
- 1-minute OHLCV
- Date range: **Feb 1–28, 2025**
- Loaded via Binance API or cache

---

## 🔍 Strategies

Implemented strategies:

1. **SMA Crossover** – Fast/slow SMA crossover with volatility filter
2. **RSI + Bollinger Bands** – Buy on RSI < 30 with BB confirmation
3. **VWAP Reversion** – Trade significant deviations from VWAP
4. **Multi-timeframe Momentum** – Combine signals from 1m and 15m
5. **ATR Trailing Breakout** – Enter on breakout, trail stop with ATR
6. **Volume Spike Breakout** – Entry on volume surge and breakout

---

## 📈 Results

Each strategy outputs:

- **Equity Curve**
- **Heatmap** of performance by symbol
- **HTML Report** with interactive charts
- **Metrics:**
  - `total_return`
  - `sharpe_ratio`
  - `max_drawdown`
  - `win_rate`
  - `exposure_time`

---

## 🛠 Project Structure

```
project/
├── core/
│   ├── data_loader.py
│   ├── backtester.py
│   └── metrics.py
├── strategies/
│   ├── base.py
│   ├── sma_cross.py
│   ├── rsi_bb.py
│   ├── vwap_reversion.py
│   ├── multi_tf_momentum.py
│   ├── atr_trailing_breakout.py
│   └── volume_spike_breakout.py
├── tests/
│   ├── test_backtester.py
│   ├── test_data_loader.py
│   └── test_strategies.py
├── data/
│   └── btc_1m_feb25.parquet
├── results/
│   ├── metrics.csv
│   ├── screenshots/
│   └── html/
├── main.py
├── requirements.txt
└── README.md
```

---

## 🔬 Testing

1. **Run all tests:**
```bash
pytest
```

2. **Test coverage:**
- `test_data_loader.py` – data caching, integrity
- `test_backtester.py` – test run_all flow
- `test_strategies.py` – 1 unit test per strategy

---

## 🌐 HTML Reports

Generated automatically:
```
results/html/SmaCrossStrategy_report.html
```
Open in your browser to explore interactive visualizations.

---

## 💡 Scalability & Extensions

- Add other exchanges (via CCXT or direct API)
- Create new strategies by extending `StrategyBase`
- Multi-timeframe and multi-asset portfolios
- Parallel backtesting (joblib, dask)
- Dashboard with Streamlit or Dash

---

## 📄 License

MIT License

---

## 🚀 Author

Developer: Oleksandr Lystopad  
- GitHub: https://github.com/l1stopad
- LinkedIn: https://www.linkedin.com/in/oleksandr-lystopad-542142218/
