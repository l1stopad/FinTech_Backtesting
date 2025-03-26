import vectorbt as vbt
import pandas as pd

def compute_metrics(pf: vbt.Portfolio) -> dict:
    total_return = pf.total_return().mean()
    sharpe_ratio = pf.sharpe_ratio().mean()
    max_drawdown = pf.max_drawdown().mean()

    # Рахуємо win_rate
    trades = pf.get_trades()
    records = trades.records_readable
    if not records.empty and "PnL" in records.columns:
        win_mask = records["PnL"] > 0
        if "Column" in records.columns:
            win_rate_by_symbol = win_mask.groupby(records["Column"]).mean()
            win_rate = win_rate_by_symbol.mean()
        else:
            win_rate = float(win_mask.mean())
    else:
        win_rate = None

    # Exposure time
    exposure_time = compute_exposure_time(pf)  # Викликає нашу функцію з об'єднанням інтервалів

    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "exposure_time": exposure_time
    }

def compute_exposure_time(pf: vbt.Portfolio) -> float:
    # (Повна реалізація, описана вище)
    trades = pf.get_trades()
    records = trades.records_readable
    if records.empty:
        return 0.0

    if "Entry Time" not in records.columns or "Exit Time" not in records.columns:
        return 0.0

    intervals = []
    for _, row in records.iterrows():
        start_t = row["Entry Time"]
        end_t = row["Exit Time"]
        if pd.isnull(end_t):
            end_t = pf.wrapper.index[-1]
        intervals.append((start_t, end_t))

    merged = unify_intervals(intervals)
    if not merged:
        return 0.0

    start_all = pf.wrapper.index[0]
    end_all = pf.wrapper.index[-1]
    total_occupied = pd.Timedelta(0)

    for (st, en) in merged:
        s_clamp = max(st, start_all)
        e_clamp = min(en, end_all)
        if e_clamp > s_clamp:
            total_occupied += (e_clamp - s_clamp)

    full_range = end_all - start_all
    if full_range <= pd.Timedelta(0):
        return 0.0

    fraction_val = total_occupied.total_seconds() / full_range.total_seconds()
    return fraction_val

def unify_intervals(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = []
    current_start, current_end = intervals[0]
    for (start, end) in intervals[1:]:
        if start <= current_end:
            if end > current_end:
                current_end = end
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged
