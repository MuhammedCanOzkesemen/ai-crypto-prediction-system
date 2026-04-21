import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from core.backtest.simulator import simulate
from core.models.classifier import CryptoClassifier
from core.decision.engine import decide

FAST_MODE = False

CANDIDATE_PATHS = [
    "data/raw/BTC.parquet",
    "data/raw/btc.parquet",
    "data/BTC.parquet",
    "data/btc.parquet",
    "data/raw/BTCUSDT.parquet",
    "data/raw/BTC.csv",
    "data/raw/btc.csv",
    "data/BTC.csv",
    "data/btc.csv",
    "data/raw/BTCUSDT.csv",
]

REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


def load_dataset() -> pd.DataFrame:
    for path in CANDIDATE_PATHS:
        try:
            if path.endswith(".parquet"):
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
            print(f"Loaded: {path}")
            return df
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"Error reading {path}: {e}")
            continue
    return None


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    features = pd.DataFrame(index=df.index)

    returns = close.pct_change()
    features["returns"] = returns
    features["log_volume"] = np.log1p(volume)
    features["hl_range"] = (high - low) / close
    features["close_open"] = (close - df["open"]) / df["open"]
    for lag in [1, 2, 3, 5]:
        features[f"ret_lag_{lag}"] = returns.shift(lag)

    vol_5d = returns.rolling(5).std()
    vol_20d = returns.rolling(20).std()
    features["realized_vol_5d"] = vol_5d
    features["return_5d"] = close.pct_change(5)

    return_3d = close.pct_change(3)
    return_7d = close.pct_change(7)
    features["momentum_accel"] = return_3d - return_7d

    features["vol_shock"] = vol_5d / vol_20d.replace(0, np.nan)

    hl_5d = (high.rolling(5).max() - low.rolling(5).min())
    hl_20d = (high.rolling(20).max() - low.rolling(20).min())
    features["range_compression"] = hl_5d / hl_20d.replace(0, np.nan)

    sma20 = close.rolling(20).mean()
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    features["trend_strength"] = (close - sma20).abs() / atr.replace(0, np.nan)

    rolling_high_10 = high.shift(1).rolling(10).max()
    features["breakout"] = (close > rolling_high_10).astype(float)

    sma10 = close.rolling(10).mean()
    std10 = close.rolling(10).std()
    features["mean_reversion"] = (close - sma10) / std10.replace(0, np.nan)

    vol_ma30 = volume.rolling(30).mean()
    features["volume_spike"] = volume / vol_ma30.replace(0, np.nan)

    features = features.dropna()
    return features


BIG_MOVE_THRESHOLD = 0.03


def build_labels(df: pd.DataFrame, features: pd.DataFrame, horizon: int = 7) -> pd.Series:
    forward_return = df["close"].pct_change(horizon).shift(-horizon)
    labels = (forward_return.abs() > BIG_MOVE_THRESHOLD).astype(int)
    return labels.loc[features.index].dropna()


def execute_trades(ohlc_df: pd.DataFrame, signals: np.ndarray, min_history: int, horizon: int) -> list[dict]:
    trades = []
    n = len(ohlc_df)
    for t in range(min_history, n - horizon - 1):
        if not signals[t]:
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
        trades.append({
            "entry_price": entry,
            "exit_price": exit_price,
            "return": (exit_price - entry) / entry,
        })
    return trades


def fast_model_signals(
    feat: pd.DataFrame,
    lbl: pd.Series,
    feat_reset: pd.DataFrame,
    vol_series: pd.Series,
    n: int,
    min_history: int,
    horizon: int,
    prob_threshold: float = 0.60,
    top_pct: float = 0.20,
    rolling_window: int = 50,
) -> np.ndarray:
    split = int(len(feat) * 0.7)
    X_train = feat.iloc[:split]
    y_train = lbl.iloc[:split]
    cal_size = max(1, int(split * 0.1))
    X_cal = feat.iloc[split - cal_size:split]
    y_cal = lbl.iloc[split - cal_size:split]

    clf = CryptoClassifier(random_seed=42)
    clf.fit(X_train, y_train)
    clf.calibrate(X_cal, y_cal)

    test_features = feat_reset.iloc[split:]
    probas = clf.predict_proba(test_features)[:, 1]

    return_3d_series = feat_reset["return_5d"].shift(2) if "return_5d" in feat_reset.columns else None
    if "ret_lag_3" in feat_reset.columns:
        return_3d_series = feat_reset["ret_lag_3"]

    signals = np.zeros(n, dtype=bool)
    recent_probs = []

    for i, t in enumerate(range(split, n - horizon - 1)):
        prob = float(probas[i])
        recent_probs.append(prob)
        window = recent_probs[-rolling_window:]
        top_cutoff = np.quantile(window, 1.0 - top_pct) if len(window) >= rolling_window else float("inf")

        if prob < prob_threshold or prob < top_cutoff:
            continue

        vol = float(vol_series.iloc[t])
        decision = decide(
            probability=prob,
            realized_vol_5d=vol,
            last_trade_day=None,
            current_day=0,
            probability_threshold=prob_threshold,
        )
        if decision["signal"] != "BUY":
            continue

        if return_3d_series is not None:
            ret3 = float(return_3d_series.iloc[t])
            if ret3 <= 0:
                continue

        signals[t] = True

    return signals


def print_results(label: str, trades: list[dict]) -> None:
    print(f"\n--- {label} ---")
    if not trades:
        print("No trades generated.")
        return
    returns = np.array([t["return"] for t in trades])
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    total_return = (1 + returns).prod() - 1
    win_rate = len(wins) / len(returns)
    gross_profit = wins.sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    print(f"Trade count:    {len(trades)}")
    print(f"Total return:   {total_return:.4%}")
    print(f"Win rate:       {win_rate:.4%}")
    print(f"Profit factor:  {profit_factor:.4f}")


def main():
    df = load_dataset()
    if df is None:
        print("ERROR: No dataset found. Searched paths:")
        for p in CANDIDATE_PATHS:
            print(f"  {p}")
        sys.exit(1)

    df.columns = [c.lower() for c in df.columns]
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        print(f"ERROR: Missing required columns: {sorted(missing)}")
        sys.exit(1)

    df = df.tail(400).reset_index(drop=True)

    print(f"Dataset rows: {len(df)}")
    print(f"Fast mode: {FAST_MODE}")

    features = build_features(df)
    ohlc = df[["open", "high", "low", "close"]]

    rng = np.random.default_rng(42)

    for horizon in [7]:
        print(f"\n{'='*40}")
        print(f"HORIZON: {horizon} days")
        print(f"{'='*40}")

        labels = build_labels(df, features, horizon=horizon)
        common_index = features.index.intersection(labels.index)
        feat = features.loc[common_index]
        lbl = labels.loc[common_index]
        ohlc_h = ohlc.loc[common_index].reset_index(drop=True)
        feat_reset = feat.reset_index(drop=True)
        vol_reset = feat_reset["realized_vol_5d"]
        n = len(ohlc_h)

        if FAST_MODE:
            model_signals = fast_model_signals(
                feat, lbl, feat_reset, vol_reset, n, min_history=200, horizon=horizon, prob_threshold=0.60
            )
            model_trades = execute_trades(ohlc_h, model_signals, min_history=200, horizon=horizon)
        else:
            model_trades = simulate(feat, lbl, ohlc_h, min_history=200, horizon=horizon, probability_threshold=0.60)

        print_results("MODEL", model_trades)

        random_signals = rng.integers(0, 2, size=n).astype(bool)
        random_trades = execute_trades(ohlc_h, random_signals, min_history=200, horizon=horizon)
        print_results("RANDOM", random_trades)

        always_buy_signals = np.ones(n, dtype=bool)
        always_buy_trades = execute_trades(ohlc_h, always_buy_signals, min_history=200, horizon=horizon)
        print_results("ALWAYS BUY", always_buy_trades)


if __name__ == "__main__":
    main()
