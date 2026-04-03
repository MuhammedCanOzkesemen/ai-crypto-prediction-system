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
| `evaluation/` | Metric computation and backtest scripts (consumes saved features / labels). |

---

## 3. Feature and Training Contract

- **Input to training:** OHLCV DataFrames; `build_features(..., coin=..., include_targets=True)` adds targets and optional Twitter fields.  
- **Schema:** `FEATURE_SCHEMA_VERSION` in `features/schema.py`; `training_metadata.json` per coin stores column order, optional `per_model_feature_indices`, historical return tails for realism checks.  
- **Targets:** Single-step **next-day log return**; multi-day **price path** is produced at inference by chaining predictions and updating synthetic OHLCV rows (`prediction/recursive_forecast.py`).

---

## 4. Prediction Pipeline (Detailed)

### 4.1 Entry point

`Predictor.predict_from_latest_features(coin)`:

1. Loads coin feature parquet (must include OHLCV columns).  
2. Loads models + scaler bundle + `training_metadata.json`.  
3. Builds `full_feat` = `build_features` on recent history tail.  
4. **Feature sanity** (`prediction/feature_sanity.py`) on the latest row.  
5. **Multi-day consensus** — three truncated histories; `_ensemble_one_step_log_return` per cutoff (`prediction/multi_day_consensus.py`).  
6. **Cooldown** read (`prediction/signal_cooldown.py`).  
7. **Market regime** — `detect_market_regime(full_feat)` (`prediction/market_regime.py`).  
8. **Recursive forecast** — 14 steps, daily caps, per-step model agreement (`prediction/realism.py`, `recursive_forecast.py`).  
9. From the path: **stability score**, **LR-matrix chaotic disagreement**, **volatility shock** vs EWMA history, **trend confirmation** (EMA20/50, ADX persistence).  
10. **Financial confidence** — `compute_financial_confidence` in `forecast_intel.py` (includes optional chaotic penalty, Twitter sentiment block).  
11. **Finalize** then **risk-adjusted confidence** — `compose_risk_adjusted_confidence` in `trading_validation.py` (regime: VOLATILE ×0.75, TRENDING ×1.05, RANGING ×0.96; caps for shock/high vol/chaos).  
12. **Decision layer** — `decision_layer.compute_decision_bundle` (legacy `trade_signal`, edge from structural formula, `confidence_after_decision`).  
13. **Trade filter** — `trade_filter.evaluate_trade_opportunity` (strict `trade_decision`, product-based `edge_score` overwrite on diagnostics).  
14. **Cooldown register** when `trade_decision` is `STRONG_BUY` or `WEAK_BUY` and cooldown not already active.

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

Response models include: regime fields, stability, consensus, trend confirmation, shock flag, Twitter sentiment diagnostics, trade decision, edge score.

---

## 9. Artifacts and Caching

- **Models / scaler:** `artifacts/models/{Coin}/`  
- **Features parquet:** configured `features_dir` (often `artifacts/features/`).  
- **Twitter cache:** under artifact root `cache/twitter_sentiment/`.  
- **Signal cooldown:** `cache/signal_cooldown/{slug}.json`.

---

## 10. Testing

`tests/` covers decision layer, forecast intel disagreement behavior, signal contracts, trade filter, trading validation helpers, and market regime classification. Run:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## 11. Design Principles

- **One-way dependencies:** `features` does not import `prediction`; `api` calls `prediction` only at the edge.  
- **Versioned schema:** Feature or target changes bump `FEATURE_SCHEMA_VERSION` and require retraining.  
- **Graceful degradation:** Missing Twitter token, failed sanity, or partial history should yield neutral or safe defaults, not crashes.

This architecture is intended to stay modular as you add backtesting harnesses or execution adapters without rewriting the core feature or model contracts.
