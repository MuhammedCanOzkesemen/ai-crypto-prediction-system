import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

from scripts.run_core_backtest_smoke import (
    load_dataset,
    build_features,
    build_labels,
    fast_model_signals,
    execute_trades,
)

HORIZONS = [3, 5, 7]
THRESHOLDS = [0.55, 0.60, 0.65]
MIN_HISTORY = 200


def compute_metrics(trades: list[dict]) -> dict:
    if not trades:
        return {"trades": 0, "total_return": 0.0, "win_rate": 0.0, "profit_factor": 0.0}
    returns = np.array([t["return"] for t in trades])
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    gross_profit = wins.sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0
    return {
        "trades": len(trades),
        "total_return": float((1 + returns).prod() - 1),
        "win_rate": float(len(wins) / len(returns)),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else float("inf"),
    }


def main():
    df = load_dataset()
    if df is None:
        print("ERROR: No dataset found.")
        sys.exit(1)

    df.columns = [c.lower() for c in df.columns]
    print(f"Dataset rows: {len(df)}\n")

    features = build_features(df)
    ohlc = df[["open", "high", "low", "close"]]
    rng = np.random.default_rng(42)

    rows = []

    for horizon in HORIZONS:
        labels = build_labels(df, features, horizon=horizon)
        common_index = features.index.intersection(labels.index)
        feat = features.loc[common_index]
        lbl = labels.loc[common_index]
        ohlc_h = ohlc.loc[common_index].reset_index(drop=True)
        feat_reset = feat.reset_index(drop=True)
        vol_reset = feat_reset["realized_vol_5d"]
        n = len(ohlc_h)

        random_signals = rng.integers(0, 2, size=n).astype(bool)
        random_metrics = compute_metrics(execute_trades(ohlc_h, random_signals, MIN_HISTORY, horizon))
        random_pf = random_metrics["profit_factor"] if random_metrics["profit_factor"] > 0 else 1.0

        for threshold in THRESHOLDS:
            signals = fast_model_signals(
                feat, lbl, feat_reset, vol_reset, n, MIN_HISTORY, horizon, prob_threshold=threshold
            )
            trades = execute_trades(ohlc_h, signals, MIN_HISTORY, horizon)
            m = compute_metrics(trades)
            pf_vs_random = m["profit_factor"] / random_pf if random_pf > 0 else float("inf")

            rows.append({
                "H": horizon,
                "TH": threshold,
                "TRADES": m["trades"],
                "RETURN": m["total_return"],
                "WIN": m["win_rate"],
                "PF": m["profit_factor"],
                "PF_vs_RANDOM": pf_vs_random,
            })

    rows.sort(key=lambda r: r["PF"], reverse=True)

    header = f"{'H':>4} | {'TH':>5} | {'TRADES':>6} | {'RETURN':>8} | {'WIN':>6} | {'PF':>6} | {'PF_vs_RANDOM':>12}"
    print(header)
    print("-" * len(header))
    for r in rows:
        pf_str = f"{r['PF']:.4f}" if r["PF"] != float("inf") else "  inf"
        pvr_str = f"{r['PF_vs_RANDOM']:.4f}" if r["PF_vs_RANDOM"] != float("inf") else "   inf"
        print(
            f"{r['H']:>4} | {r['TH']:>5.2f} | {r['TRADES']:>6} | "
            f"{r['RETURN']:>8.4%} | {r['WIN']:>6.4%} | {pf_str:>6} | {pvr_str:>12}"
        )


if __name__ == "__main__":
    main()
