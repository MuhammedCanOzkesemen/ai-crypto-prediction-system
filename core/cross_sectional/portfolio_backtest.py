import numpy as np
import pandas as pd

from core.cross_sectional.ranker import CrossSectionalRanker

SLIPPAGE = 0.001


def simulate_portfolio(
    features_by_coin: dict[str, pd.DataFrame],
    ohlc_by_coin: dict[str, pd.DataFrame],
    ranker: CrossSectionalRanker,
    test_dates: list,
    top_n: int = 2,
) -> list[dict]:
    sorted_dates = sorted(test_dates)
    date_index = {d: i for i, d in enumerate(sorted_dates)}
    trades = []

    for i, date in enumerate(sorted_dates[:-1]):
        next_date = sorted_dates[i + 1]

        feature_rows = {}
        for coin, feat_df in features_by_coin.items():
            if date in feat_df.index:
                feature_rows[coin] = feat_df.loc[[date]]

        if len(feature_rows) < 2:
            continue

        ranked = ranker.rank_date(feature_rows)
        top_coins = [coin for coin, _ in ranked[:top_n]]

        coin_returns = []
        selected = []
        for coin in top_coins:
            ohlc = ohlc_by_coin[coin]
            if next_date not in ohlc.index:
                continue
            entry = float(ohlc.loc[next_date, "open"]) * (1 + SLIPPAGE)
            exit_ = float(ohlc.loc[next_date, "close"]) * (1 - SLIPPAGE)
            if entry > 0:
                coin_returns.append((exit_ - entry) / entry)
                selected.append(coin)

        if not coin_returns:
            continue

        trades.append({
            "date": date,
            "coins": selected,
            "return": float(np.mean(coin_returns)),
        })

    return trades


def compute_portfolio_metrics(trades: list[dict]) -> dict:
    if not trades:
        return {}

    returns = np.array([t["return"] for t in trades])
    wins = returns[returns > 0]
    losses = returns[returns <= 0]

    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = float(drawdowns.min())

    total_return = float(cumulative[-1] - 1)
    win_rate = float(len(wins) / len(returns))
    gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_return = float(returns.mean())
    std_return = float(returns.std())
    sharpe = float((avg_return / std_return) * np.sqrt(252)) if std_return > 0 else 0.0

    return {
        "trade_count": len(trades),
        "total_return": total_return,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
    }
