"""
Long-only trade simulation on daily OHLC bars.

Assumptions (daily data only):
- Entry at the session open of the bar after the decision date.
- Intraday path: if both stop and take-profit lie inside [low, high], we assume stop loss
  triggers first (conservative for longs).
- If the open gaps through a level, we exit at the open when it is at or beyond that level.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def simulate_long_trade_daily_ohlc(
    ohlc: pd.DataFrame,
    entry_row_index: int,
    *,
    stop_loss_pct: float,
    take_profit_pct: float,
    max_holding_bars: int,
) -> dict[str, Any]:
    """
    Simulate a long opened at ``open`` of ``entry_row_index``.

    ``ohlc`` must be sorted by time and include columns open, high, low, close.

    Returns dict with exit_price, exit_reason (TP|SL|TIME|INVALID), return_pct, holding_days,
    exit_row_index, entry_price.
    """
    if ohlc is None or ohlc.empty or entry_row_index < 0 or entry_row_index >= len(ohlc):
        return {
            "entry_price": None,
            "exit_price": None,
            "return_pct": None,
            "holding_days": None,
            "exit_reason": "INVALID",
            "exit_row_index": None,
        }

    entry_open = float(ohlc.iloc[entry_row_index]["open"])
    if entry_open <= 0 or not (entry_open == entry_open):
        return {
            "entry_price": None,
            "exit_price": None,
            "return_pct": None,
            "holding_days": None,
            "exit_reason": "INVALID",
            "exit_row_index": None,
        }

    sl_price = entry_open * (1.0 - float(stop_loss_pct))
    tp_price = entry_open * (1.0 + float(take_profit_pct))
    max_h = max(1, int(max_holding_bars))

    for j in range(max_h):
        i = entry_row_index + j
        if i >= len(ohlc):
            break
        row = ohlc.iloc[i]
        o = float(row["open"])
        hi = float(row["high"])
        lo = float(row["low"])
        cl = float(row["close"])

        if j > 0:
            if o <= sl_price:
                ret = (o / entry_open) - 1.0
                return {
                    "entry_price": entry_open,
                    "exit_price": o,
                    "return_pct": float(ret),
                    "holding_days": j + 1,
                    "exit_reason": "SL",
                    "exit_row_index": i,
                }
            if o >= tp_price:
                ret = (o / entry_open) - 1.0
                return {
                    "entry_price": entry_open,
                    "exit_price": o,
                    "return_pct": float(ret),
                    "holding_days": j + 1,
                    "exit_reason": "TP",
                    "exit_row_index": i,
                }

        if j == 0:
            if o <= sl_price:
                ret = (o / entry_open) - 1.0
                return {
                    "entry_price": entry_open,
                    "exit_price": o,
                    "return_pct": ret,
                    "holding_days": j + 1,
                    "exit_reason": "SL",
                    "exit_row_index": i,
                }
            if o >= tp_price:
                ret = (o / entry_open) - 1.0
                return {
                    "entry_price": entry_open,
                    "exit_price": o,
                    "return_pct": ret,
                    "holding_days": j + 1,
                    "exit_reason": "TP",
                    "exit_row_index": i,
                }

        touched_sl = lo <= sl_price
        touched_tp = hi >= tp_price
        if touched_sl and touched_tp:
            exit_px = sl_price
            reason = "SL"
        elif touched_sl:
            exit_px = sl_price
            reason = "SL"
        elif touched_tp:
            exit_px = tp_price
            reason = "TP"
        else:
            exit_px = None
            reason = None

        if reason:
            ret = (exit_px / entry_open) - 1.0
            return {
                "entry_price": entry_open,
                "exit_price": float(exit_px),
                "return_pct": float(ret),
                "holding_days": j + 1,
                "exit_reason": reason,
                "exit_row_index": i,
            }

        if j == max_h - 1:
            ret = (cl / entry_open) - 1.0
            return {
                "entry_price": entry_open,
                "exit_price": cl,
                "return_pct": float(ret),
                "holding_days": j + 1,
                "exit_reason": "TIME",
                "exit_row_index": i,
            }

    return {
        "entry_price": entry_open,
        "exit_price": None,
        "return_pct": None,
        "holding_days": None,
        "exit_reason": "INVALID",
        "exit_row_index": None,
    }
