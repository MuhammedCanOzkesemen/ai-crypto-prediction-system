# Architecture — AI Crypto Trading Intelligence System

This document describes how the repository is structured, how data moves through the stack, and how validation and trade decisions are composed. It reflects the **current** implementation (ensemble regressors on tabular features, recursive horizon forecast, FastAPI, static dashboard).

---

## 1. High-Level Data Flow

```
[data/price_fetcher, data_refresh]  →  OHLCV (+ stored parquet features)
        ↓
[features/feature_builder]  →  indicators, targets, optional Twitter columns
        ↓
[scripts/train_pipeline]  →  scaled matrices, RF/XGB/LGBM artifacts, metadata
        ↓
[prediction/predictor]  →  recursive path, confidence, regime, trade tiers
        ↓
[api/app.py]  →  JSON responses
        ↓
[dashboard/]  →  static HTML/JS/CSS
```

Supporting cross-cutting packages: **`utils/`** (config, constants, logging), **`evaluation/`** (metrics, backtest helpers), **`tests/`** (unit tests).

---

## 2. Directory and Module Map

| Path | Role |
|------|------|
| `data/` | Historical fetch, refresh orchestration, optional `twitter_sentiment` integration points. |
| `features/` | `feature_builder.py` builds ML-ready frames; `schema.py` versions the training column contract. |
| `models/` | Wrappers for Random Forest, XGBoost, LightGBM; `feature_subsets.py` per-model column views; registry for discovery. |
| `prediction/` | Inference, path recursion, intelligence, validation, regime, trade filter, cooldown. See §4–§6. |
| `api/` | FastAPI app, Pydantic response models, routes under `/predictions`, `/api/*`, static dashboard mount. |
| `dashboard/` | Client UI: forecast table, chart (Plotly), regime, risk strip, trade signal. |
| `scripts/train_pipeline.py` | End-to-end train: fetch → features → fit → evaluate → write artifacts. |
| `evaluation/` | Model metrics, walk-forward **model** backtest (`backtest.py`), **trade** backtest (`trade_backtest.py`), execution sim (`trade_execution.py`), metrics (`backtest_metrics.py`), ablation (`ablation.py`). |

---

## 3. Feature and Training Contract

- **Input to training:** OHLCV DataFrames; `build_features(..., coin=..., include_targets=True)` adds targets and optional Twitter fields.  
- **Schema:** `FEATURE_SCHEMA_VERSION` in `features/schema.py`; `training_metadata.json` per coin stores column order, optional `per_model_feature_indices`, historical return tails for realism checks.  
- **Targets:** Single-step **next-day log return**; multi-day **price path** is produced at inference by chaining predictions and updating synthetic OHLCV rows (`prediction/recursive_forecast.py`).

---

## 4. Prediction Pipeline (Detailed)

### 4.1 Entry point

`Predictor.predict_from_latest_features(coin)` delegates to **`Predictor._run_full_forecast_pipeline`** with live defaults (Twitter on, regime gating on, disk cooldown, strict trade filter, `decision_gate_multiplier=1.0`). **`Predictor.predict_for_backtest`** calls the same core with overrides for research (optional neutral Twitter via `coin=None` in `build_features`, optional neutral regime for gating, simulated cooldown flag, no disk cooldown write, optional `decision_mode` for weaker gates or decision-layer-only tiers).

Pipeline steps:

1. Loads coin feature parquet (must include OHLCV columns) — or receives an OHLCV-only frame in backtest mode; feature context and directional head use `full_feat` when present.  
2. Loads models + scaler bundle + `training_metadata.json`.  
3. Builds `full_feat` = `build_features` on recent history tail (`coin=None` disables Twitter merge for ablations).  
4. **Feature sanity** (`prediction/feature_sanity.py`) on the latest row.  
5. **Multi-day consensus** — three truncated histories; `_ensemble_one_step_log_return` per cutoff (`prediction/multi_day_consensus.py`).  
6. **Cooldown** — live: `signal_cooldown.cooldown_active`; backtest: boolean override per simulated day.  
7. **Market regime** — `detect_market_regime(full_feat)`; diagnostics always store detected regime; gating for confidence/trade filter can be forced to `RANGING` when `use_regime_gating=False`.  
8. **Recursive forecast** — 14 steps, daily caps, per-step model agreement (`prediction/realism.py`, `recursive_forecast.py`).  
9. From the path: **stability score**, **LR-matrix chaotic disagreement**, **volatility shock** vs EWMA history, **trend confirmation** (EMA20/50, ADX persistence).  
10. **Financial confidence** — `compute_financial_confidence` in `forecast_intel.py` (includes optional chaotic penalty, Twitter sentiment block).  
11. **Finalize** then **risk-adjusted confidence** — `compose_risk_adjusted_confidence` in `trading_validation.py` (effective regime for multipliers follows gating; caps for shock/high vol/chaos).  
12. **Decision layer** — `decision_layer.compute_decision_bundle` (optional `decision_gate_multiplier` for relaxed research gates).  
13. **Trade filter** — `trade_filter.evaluate_trade_opportunity` unless `strict_trade_filter=False` (maps `trade_signal` BUY to strong/weak tier for ablation F).  
14. **Cooldown register** (live only) when `trade_decision` is actionable — backtests set `register_signal_cooldown=False` and simulate cooldown in the outer loop.

### 4.2 Recursive forecast

- Each step: rebuild features from rolling synthetic + real history, align to scaler, slice per model if metadata specifies indices, predict clipped log return, update prices.  
- Diagnostics: clip saturation, emergency align flags, path variance signals (`prediction/forecast_path_quality.py`).

---

## 5. Market Regime and Adaptive Behavior

**Module:** `prediction/market_regime.py`

- **VOLATILE** — Current EWMA log-vol vs recent window shows a spike (default ratio ≥ ~1.75× trailing values).  
- **TRENDING** — ADX above threshold and trend consistency supports a directional market (not volatile-first).  
- **RANGING** — Low ADX and mid range position, or default transition.

**Outputs:** `market_regime`, `market_regime_confidence` (0–1).

**Confidence:** `compose_risk_adjusted_confidence` applies regime multipliers after the core product.

**Trade filter:**

- **TRENDING** — Slightly lower agreement bar for strong tier; slightly relaxed consensus/stability/trend thresholds for strong layers.  
- **RANGING** — **STRONG_BUY** disabled; **WEAK_BUY** requires tighter confidence, agreement, stability, consensus, and move.  
- **VOLATILE** — Edge score ×0.7; **NO_TRADE** unless an **extreme** confidence bundle is met; only narrow **WEAK_BUY** if that gate passes.

---

## 6. Validation Layers (Summary)

| Layer | Source | Effect |
|-------|--------|--------|
| Feature sanity | Latest row audit | Forces degraded input + trade NO_TRADE path |
| Multi-day consensus | 3× single-step directions | Score + reasons; trade thresholds |
| Path stability | Variance + sign flips along 14d path | Risk-adjusted confidence + weak gates |
| Trend confirmation | EMA cross + ADX persistence | Risk-adjusted confidence + strong layers |
| Volatility shock | EWMA vs history | Shock flag, STRONG block, confidence cap |
| Chaotic disagreement | Sign split across models per day | Extra financial confidence penalty + cap 0.5 |
| Regime | ADX / vol / structure | Confidence multipliers + adaptive trade rules |
| Cooldown | Disk JSON per coin | NO_TRADE with reason while active |

---

## 7. Decision Logic (Two Tracks)

1. **Legacy decision layer** — Uses expected move strength, R:R, alignment, adaptive thresholds; produces `trade_signal` and structural `edge_score` before trade filter overwrites displayed edge.  
2. **Strict trade engine** — `STRONG_BUY` / `WEAK_BUY` / `NO_TRADE` with regime-aware rules; `trade_reasons` explain failures and downgrades.

Consumers should treat **`trade_decision`** as the primary strict label for research and UI; **`trade_signal`** remains for backward compatibility.

---

## 8. API Surface

- `GET /predictions/{coin}` — Compact prediction + diagnostics subset.  
- `GET /api/forecast-path/{coin}` — Full path, multi-horizon summary, extended diagnostics.  
- `POST /api/refresh/{coin}` — Force OHLCV + feature rebuild.  
- `GET /api/chart/{coin}`, `GET /api/evaluation/{coin}` — Chart and metric summaries.  
- `GET /api/backtest/{coin}` — Latest saved **`artifacts/backtests/{Coin}_summary.json`** from an offline `scripts/run_backtest.py` run (404 if missing).

Response models include: regime fields, stability, consensus, trend confirmation, shock flag, Twitter sentiment diagnostics, trade decision, edge score.

---

## 9. Artifacts and Caching

- **Models / scaler:** `artifacts/models/{Coin}/`  
- **Features parquet:** configured `features_dir` (often `artifacts/features/`).  
- **Twitter cache:** under artifact root `cache/twitter_sentiment/`.  
- **Signal cooldown:** `cache/signal_cooldown/{slug}.json`.  
- **Trade backtests:** `backtests/{Coin}_trades.csv`, `{Coin}_equity_curve.csv`, `{Coin}_summary.json`, optional `{Coin}_ablation.csv`, `backtest_leaderboard.csv`.

---

## 10. Trade Backtest and Ablation (Research)

**Modules:** `evaluation/trade_backtest.py`, `evaluation/trade_execution.py`, `evaluation/backtest_metrics.py`, `evaluation/ablation.py`, CLI `scripts/run_backtest.py`.

**No leakage:** For each simulated calendar day `t`, the predictor receives `ohlcv.iloc[: t+1]` only. Twitter-off ablations use `build_features(..., coin=None)` so sentiment columns are neutral for that slice. Cooldown in simulation does not read the live JSON file when the override is supplied.

**Execution (daily bars):** Entry at the **next** bar’s **open** after a `STRONG_BUY` / `WEAK_BUY`. STRONG: stop −2%, take-profit +3%, up to 3 sessions. WEAK: stop −1.5%, take-profit +2%, 1 session. Intraday touch uses `high`/`low`; if both stop and profit levels lie inside the same bar’s range, **stop is assumed first** (conservative long). Gap opens through a level exit at **open**.

**Metrics:** Trade list P&L, compounded equity, drawdown on the daily equity series, Sharpe/Sortino on daily simple returns, profit factor, regime-stratified returns, and health counters (signal frequency, cooldown days, coarse rejection buckets from text reasons).

**Ablation:** Six rows (A–F) written to `{Coin}_ablation.csv` with aligned columns for quick comparison.

---

## 11. Testing

`tests/` covers decision layer, forecast intel disagreement behavior, signal contracts, trade filter, trading validation helpers, market regime classification, and trade execution / ablation CSV shape. Run:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## 12. Design Principles

- **One-way dependencies:** `features` does not import `prediction`; `api` calls `prediction` only at the edge.  
- **Versioned schema:** Feature or target changes bump `FEATURE_SCHEMA_VERSION` and require retraining.  
- **Graceful degradation:** Missing Twitter token, failed sanity, or partial history should yield neutral or safe defaults, not crashes.

This architecture is intended to stay modular as you add backtesting harnesses or execution adapters without rewriting the core feature or model contracts.
