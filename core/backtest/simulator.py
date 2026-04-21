import pandas as pd

from core.live.predictor import run_prediction


def simulate(
    features_df: pd.DataFrame,
    labels: pd.Series,
    ohlc_df: pd.DataFrame,
    min_history: int = 200,
    horizon: int = 3,
    probability_threshold: float = 0.58,
) -> list[dict]:
    max_steps = 150
    trades = []
    n = len(features_df)

    for t in range(min_history, min(n - horizon - 1, min_history + max_steps)):
        result = run_prediction(features_df.iloc[:t], labels.iloc[:t], probability_threshold=probability_threshold)

        if result["signal"] != "BUY":
            continue

        entry = ohlc_df["open"].iloc[t + 1] * 1.001
        stop = entry * 0.98
        target = entry * 1.03
        exit_price = None

        for day in range(t + 1, t + horizon + 1):
            if ohlc_df["low"].iloc[day] <= stop:
                exit_price = stop
                break
            if ohlc_df["high"].iloc[day] >= target:
                exit_price = target
                break

        if exit_price is None:
            exit_price = ohlc_df["close"].iloc[t + horizon]

        exit_price *= 0.999
        trade_return = (exit_price - entry) / entry

        trades.append({
            "entry_price": entry,
            "exit_price": exit_price,
            "return": trade_return,
        })

    return trades
