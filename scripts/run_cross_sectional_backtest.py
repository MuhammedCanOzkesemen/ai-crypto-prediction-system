import sys
import pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from core.cross_sectional.features import build_features
from core.cross_sectional.labels import build_cross_sectional_labels
from core.cross_sectional.ranker import CrossSectionalRanker
from core.cross_sectional.portfolio_backtest import simulate_portfolio, compute_portfolio_metrics

COINS = ["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE"]
DATA_DIR = pathlib.Path("data/raw")
HORIZON = 5
TRAIN_RATIO = 0.70
CAL_RATIO = 0.10
TOP_N = 2


def load_coin(name: str) -> pd.DataFrame | None:
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        print(f"Missing: {path}")
        return None
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df.columns = [c.lower() for c in df.columns]
    return df.sort_index()


def main():
    raw = {}
    for coin in COINS:
        df = load_coin(coin)
        if df is not None:
            raw[coin] = df

    if len(raw) < 2:
        print("ERROR: Need at least 2 coins.")
        sys.exit(1)

    print(f"Loaded coins: {list(raw.keys())}")

    features_by_coin = {}
    for coin, df in raw.items():
        feat = build_features(df)
        if len(feat) > 0:
            features_by_coin[coin] = feat

    common_dates = sorted(
        set.intersection(*[set(f.index) for f in features_by_coin.values()])
    )

    if len(common_dates) < 100:
        print(f"ERROR: Only {len(common_dates)} common dates. Need more data.")
        sys.exit(1)

    print(f"Common dates: {len(common_dates)}  ({common_dates[0].date()} → {common_dates[-1].date()})")

    all_close = pd.DataFrame(
        {coin: raw[coin]["close"] for coin in features_by_coin},
        index=common_dates,
    )
    labels_df = build_cross_sectional_labels(all_close, horizon=HORIZON)
    label_dates = [d for d in common_dates if d in labels_df.index]

    n = len(label_dates)
    train_end = int(n * TRAIN_RATIO)
    cal_end = int(n * (TRAIN_RATIO + CAL_RATIO))

    train_dates = label_dates[:train_end]
    cal_dates = label_dates[train_end:cal_end]
    test_dates = label_dates[cal_end:]

    print(f"Train: {len(train_dates)}  Cal: {len(cal_dates)}  Test: {len(test_dates)}")

    def pool(dates):
        rows, ys = [], []
        for date in dates:
            for coin, feat_df in features_by_coin.items():
                if date in feat_df.index and date in labels_df.index:
                    rows.append(feat_df.loc[date])
                    ys.append(int(labels_df.loc[date, coin]))
        return pd.DataFrame(rows), pd.Series(ys, dtype=int)

    X_train, y_train = pool(train_dates)
    X_cal, y_cal = pool(cal_dates)

    print(f"Training samples: {len(X_train)}  (pos rate: {y_train.mean():.2%})")

    ranker = CrossSectionalRanker(random_seed=42)
    ranker.fit(X_train, y_train)
    ranker.calibrate(X_cal, y_cal)

    ohlc_by_coin = {
        coin: raw[coin][["open", "close"]].loc[test_dates[0]:test_dates[-1]]
        for coin in features_by_coin
    }

    trades = simulate_portfolio(
        features_by_coin=features_by_coin,
        ohlc_by_coin=ohlc_by_coin,
        ranker=ranker,
        test_dates=test_dates,
        top_n=TOP_N,
    )

    metrics = compute_portfolio_metrics(trades)

    print(f"\n{'='*40}")
    print(f"CROSS-SECTIONAL BACKTEST RESULTS")
    print(f"Coins: {list(features_by_coin.keys())}  |  Top {TOP_N} per day  |  Horizon label: {HORIZON}d")
    print(f"{'='*40}")

    if not metrics:
        print("No trades generated.")
        sys.exit(0)

    print(f"Trade count:    {metrics['trade_count']}")
    print(f"Total return:   {metrics['total_return']:.4%}")
    print(f"Win rate:       {metrics['win_rate']:.4%}")
    print(f"Profit factor:  {metrics['profit_factor']:.4f}")
    print(f"Sharpe ratio:   {metrics['sharpe_ratio']:.4f}")
    print(f"Max drawdown:   {metrics['max_drawdown']:.4%}")

    print(f"\n--- RANDOM BASELINE (equal-weight all coins) ---")
    rng = np.random.default_rng(42)
    rand_trades = []
    all_test = sorted(test_dates)
    for i, date in enumerate(all_test[:-1]):
        next_date = all_test[i + 1]
        coin_returns = []
        for coin in features_by_coin:
            ohlc = raw[coin]
            if next_date in ohlc.index:
                entry = float(ohlc.loc[next_date, "open"]) * 1.001
                exit_ = float(ohlc.loc[next_date, "close"]) * 0.999
                if entry > 0:
                    coin_returns.append((exit_ - entry) / entry)
        if coin_returns:
            chosen = rng.choice(coin_returns, size=min(TOP_N, len(coin_returns)), replace=False)
            rand_trades.append({"date": date, "return": float(np.mean(chosen))})

    rand_metrics = compute_portfolio_metrics(rand_trades)
    print(f"Trade count:    {rand_metrics['trade_count']}")
    print(f"Total return:   {rand_metrics['total_return']:.4%}")
    print(f"Win rate:       {rand_metrics['win_rate']:.4%}")
    print(f"Profit factor:  {rand_metrics['profit_factor']:.4f}")
    print(f"Sharpe ratio:   {rand_metrics['sharpe_ratio']:.4f}")


if __name__ == "__main__":
    main()
