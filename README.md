# AI-Powered Crypto Trading Intelligence System

Production-oriented cryptocurrency analytics stack that combines **multi-model forecasting**, **risk-adjusted confidence**, **multi-layer validation**, and an **adaptive trade decision engine**. It is designed to reduce noise and surface higher-conviction setups—not to replace human judgment or broker execution.

---

## What It Does

The system goes beyond raw price prediction:

- **Decision engine** — Classifies opportunities into `STRONG_BUY`, `WEAK_BUY`, and `NO_TRADE` using strict, auditable rules.
- **Risk filtering** — Stability of the forecast path, multi-day directional consensus, trend confirmation, volatility shock detection, and optional feature sanity checks feed into confidence and gating.
- **Adaptive behavior** — A **market regime** classifier (`TRENDING`, `RANGING`, `VOLATILE`) adjusts confidence multipliers and trade thresholds so logic matches the current context.
- **API and dashboard** — Structured JSON for integrations plus a browser dashboard for operators.

This is **not** automated live trading out of the box: there is no order router, position sizing, or exchange execution layer.

---

## Core Features

| Area | Description |
|------|-------------|
| **Ensemble** | Random Forest, XGBoost, LightGBM on a shared feature schema; metric-weighted combination; optional directional classifier probabilities. |
| **Sentiment** | Optional Twitter/X recent search with VADER compound scores, cached; merges into features and confidence diagnostics when volume is sufficient. |
| **Multi-layer validation** | Path stability score, 3-day ensemble consensus, EMA/ADX trend confirmation, chaotic vs directional model disagreement handling. |
| **Market regime** | `TRENDING` / `RANGING` / `VOLATILE` from ADX, EWMA log-vol structure, trend consistency, and range position; `market_regime_confidence` in `[0,1]`. |
| **Trade engine** | Regime-aware thresholds: e.g. ranging markets block `STRONG_BUY`; volatile regimes require extreme confidence for any scalp. |
| **Confidence** | Financial confidence from path metrics, then **risk-adjusted** composition (agreement, stability, consensus, trend confirm) with regime multipliers and hard caps under shock/chaos/high vol. |
| **Volatility shock** | Flags when short-term EWMA vol materially exceeds recent history (default ratio threshold ~1.8×). |
| **Signal cooldown** | Per-coin cooldown file after actionable tiers to limit over-trading (`TRADE_COOLDOWN_DAYS`, default 3). |
| **Trade backtest** | Walk-forward simulation with next-bar entry, SL/TP/time exits, CSV trade log, equity curve, regime breakdown, strategy health metrics. |
| **Ablation lab** | Six fixed variants (full / no Twitter / no regime gating / no cooldown / weaker gates / decision-layer-only) for comparable CSV summaries. |

---

## System Architecture

End-to-end layers:

```
Data → Features → Models → Validation → Decision → API → Dashboard
```

1. **Data** — Historical OHLCV ingestion (e.g. CoinGecko / Binance paths), refresh, feature parquet per coin.  
2. **Features** — Technical indicators, log-return momentum, interactions, optional Twitter aggregates.  
3. **Models** — Per-coin trained regressors (next-day log return); scaler + metadata on disk.  
4. **Validation** — Recursive 14-step path, stability/consensus/shock/trend checks, regime detection.  
5. **Decision** — Legacy BUY/SELL/NO_TRADE filter plus strict `trade_decision` tiers and edge score.  
6. **API** — FastAPI: `/predictions/{coin}`, `/api/forecast-path/{coin}`, health, refresh, chart, evaluation, saved `/api/backtest/{coin}`.  
7. **Dashboard** — Static UI: forecast path, regime, risk indicators, trade signal block.  
8. **Research** — Offline `scripts/run_backtest.py` writes `artifacts/backtests/*` (trades, equity, summary JSON, optional ablation and leaderboard).

See **ARCHITECTURE.md** for module-level detail and data flow.

---

## How It Works (Step Flow)

1. Load latest **feature parquet** (OHLCV + engineered columns) for the coin.  
2. Run **feature sanity** on the latest row (NaN / non-finite / extreme magnitudes).  
3. Build **multi-day consensus** from three historical cutoffs (single-step ensemble direction + agreement).  
4. Classify **market regime** from recent feature rows (vol spike → `VOLATILE`, else ADX/trend structure).  
5. Run **recursive forecast** (14 daily steps, vol-aware caps, model agreement per step).  
6. Compute **stability**, **chaotic disagreement**, **volatility shock**, **trend confirmation**.  
7. Derive **raw financial confidence**, apply penalties, **finalize**, then **risk-adjusted** confidence (regime multipliers + caps).  
8. Run **decision layer** (expected move, R:R, alignment) and **trade filter** (regime-adaptive `STRONG_BUY` / `WEAK_BUY` / `NO_TRADE`).  
9. If actionable and not in cooldown, **register cooldown** window.  
10. Return structured payload to API/dashboard.

---

## Trade Engine Logic (Strict Tiers)

- **`STRONG_BUY`** — High confidence, agreement, signal strength, classifier skew, move size, R:R, bullish forecast, multi-layer scores, **no** volatility shock; **never** in `RANGING` or `VOLATILE`; in `TRENDING`, agreement and layer thresholds are slightly relaxed.  
- **`WEAK_BUY`** — Lower bar for scalp-style setups; **tighter** conditions in `RANGING`; in `VOLATILE`, only after an **extreme-confidence** gate and stricter move/R:R/stability.  
- **`NO_TRADE`** — Default when any hard filter, cooldown, sanity failure, volatile gate, or soft gate fails.

Legacy **`trade_signal`** (`BUY` / `SELL` / `NO_TRADE`) from the older filter remains for compatibility; **`trade_decision`** is the strict engine output.

---

## Why This System Is Different

- **Noise reduction** — Multiple independent checks (path, models, time, regime) must align before strong labels.  
- **Bad-trade filtering** — Shocks, chaotic cross-model disagreement, low consensus, and unstable paths pull down confidence and block tiers.  
- **Context awareness** — Regime detection changes both **confidence math** and **trade rules**, closer to discretionary risk practice than a single static threshold.

---

## Example API Output (Illustrative)

```json
{
  "coin": "Bitcoin",
  "average_prediction": 98542.12,
  "lower_bound": 93200.0,
  "upper_bound": 104200.0,
  "model_agreement_score": 0.78,
  "confidence_score": 0.52,
  "mean_path_agreement": 0.74,
  "market_regime": "TRENDING",
  "market_regime_confidence": 0.71,
  "stability_score": 0.62,
  "consensus_score": 0.68,
  "trend_confirmation_score": 0.59,
  "volatility_shock_detected": false,
  "trade_decision": "WEAK_BUY",
  "trade_signal": "NO_TRADE",
  "edge_score": 0.214,
  "forecast_quality": "production",
  "directional_probabilities": { "down": 0.12, "neutral": 0.28, "up": 0.6 }
}
```

Field names and shapes match the live **Pydantic** response models; values vary with market and artifact state.

---

## Install and Run

**Requirements:** Python 3.10+, dependencies in `requirements.txt`.

```bash
cd /path/to/Bitcoin
pip install -r requirements.txt
cp .env.example .env
# Set COINGECKO_API_KEY and optional TWITTER_BEARER_TOKEN / TRADE_COOLDOWN_DAYS
```

**Train (per coin or all):**

```bash
python scripts/train_pipeline.py --coin Bitcoin
# or
python scripts/train_pipeline.py
```

**API server:**

```bash
python -m uvicorn api.app:app --host 127.0.0.1 --port 8010
```

- OpenAPI: `http://127.0.0.1:8010/docs`  
- Dashboard: `http://127.0.0.1:8010/dashboard`

**Refresh data/features for one coin:**

```bash
curl -X POST http://127.0.0.1:8010/api/refresh/Bitcoin
```

**Trade backtest (offline; requires trained models + feature parquet):**

```bash
python scripts/run_backtest.py --coin Bitcoin
python scripts/run_backtest.py --coin PEPE --start 2024-01-01 --end 2025-12-31
python scripts/run_backtest.py --all
python scripts/run_backtest.py --coin Bitcoin --ablation
```

Artifacts (default under `artifacts/backtests/`): `{Coin}_trades.csv`, `{Coin}_equity_curve.csv`, `{Coin}_summary.json`, optional `{Coin}_ablation.csv`, `backtest_leaderboard.csv` for `--all`.

After a backtest, the API can serve the saved summary (no on-request simulation):

```bash
curl http://127.0.0.1:8010/api/backtest/Bitcoin
```

---

## Backtesting, Evaluation, and Ablation

- **Engine** — `evaluation/trade_backtest.py` runs `Predictor.predict_for_backtest` day by day on history truncated to each decision date (no future feature rows). Execution uses post-entry OHLC only (realized path).  
- **Metrics** — Win rate, compounded return, drawdown, Sharpe/Sortino on daily equity returns, profit factor, expectancy, exposure, per-regime P&L breakdown, and NO_TRADE diagnostics (cooldown share, rejection buckets).  
- **Ablation** — `evaluation/ablation.py` runs variants A–F into `{Coin}_ablation.csv` for side-by-side research.

See **ARCHITECTURE.md** for assumptions (daily bars, stop-first rule when both levels trade through).

---

## Future Work

- **Execution realism** — Intraday paths, fees, slippage, and partial fills beyond daily OHLC.  
- **Live trading** — Execution adapter, order management, and portfolio risk (not included here).  
- **Reinforcement learning** — Optional policy layer on top of features and regime state (research-grade extension).

---

## Security and Configuration

- Secrets live in **environment variables** (see `.env.example`); do not commit `.env`.  
- Artifacts (models, features, cache, cooldown files) default under the configured **artifacts** directory.

---

## License

See repository **LICENSE** (e.g. MIT if stated in the project root).
