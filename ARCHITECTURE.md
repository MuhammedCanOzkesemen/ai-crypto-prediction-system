# Cryptocurrency Prediction Platform — Architecture & Module Design

## 1. High-Level Overview

The system is a **modular prediction pipeline**: raw data is ingested → features are engineered → multiple ML models train and predict → an ensemble produces final outputs → an API serves results → a dashboard visualizes them.

```
[Data Sources] → data/ → features/ → models/ → prediction/ → api/ → dashboard/
                     ↑         ↑         ↑          ↑
                   utils/ ←────────────────────────────
```

---

## 2. Folder Structure

```
Bitcoin/
├── data/                    # Data ingestion, storage, schemas
├── features/                # Feature engineering & technical indicators
├── models/                  # LSTM, RandomForest, XGBoost implementations
├── prediction/              # Ensemble logic & prediction orchestration
├── api/                     # REST/FastAPI service for predictions
├── dashboard/               # Frontend (e.g. React/Streamlit) for visualization
├── utils/                   # Shared helpers, config, logging
├── config/                  # Environment & model configs (optional; can live in utils)
├── tests/                   # Unit and integration tests
├── scripts/                 # One-off or CLI scripts (e.g. train, backtest)
├── ARCHITECTURE.md          # This document
└── README.md
```

---

## 3. Module Responsibilities

### 3.1 `data/`

**Purpose:** Single source of truth for all ingested and stored data.

- **Ingestion**
  - Fetch historical OHLCV (and volume) for the top 10 coins from a provider (e.g. CoinGecko, Binance, CryptoCompare).
  - Optional: news headlines/API and social metrics (e.g. Reddit/Twitter APIs or third-party sentiment APIs).
- **Storage**
  - Raw series (e.g. CSV/Parquet or SQLite/Postgres) with clear schema: symbol, timestamp, open, high, low, close, volume.
  - Separate tables or files for news/social if used.
- **Contracts**
  - Define canonical coin list (Bitcoin, Ethereum, BNB, XRP, Solana, Dogecoin, Cardano, Pepe, Polkadot, Chainlink) and date ranges.
- **Submodules (conceptual)**
  - `sources/`: adapters per provider (e.g. `coingecko.py`, `news_api.py`).
  - `storage/`: read/write to local DB or files.
  - `schemas/`: data models or table definitions for raw and validated data.

**Outcome:** Clean, timestamped series and optional sentiment datasets ready for feature engineering.

---

### 3.2 `features/`

**Purpose:** Transform raw data into model-ready feature matrices and targets.

- **Technical indicators (per coin)**
  - RSI, MACD (and signal/histogram), Bollinger Bands (upper/lower/width), EMA (e.g. 12/26/50), and volatility (e.g. ATR, rolling std).
- **Targets**
  - Next 1-day and multi-day (e.g. 14-day) price/return; binary “upward movement” for probability; optional volatility for ranges.
- **Sentiment (if available)**
  - Aggregate news/social sentiment into daily (or sub-daily) scores and merge with price index.
- **Pipeline**
  - One entry point that: loads data from `data/` → computes all indicators → builds train/validation/test windows → outputs DataFrames or arrays with aligned features and targets for 14-day horizon.

**Outcome:** Feature matrices and target vectors (and metadata for date/symbol) consumed by `models/` and `prediction/`.

---

### 3.3 `models/`

**Purpose:** Implement and train the three model families; expose a unified “predict” interface.

- **LSTM**
  - Time-series model (e.g. PyTorch/TensorFlow or Keras): sequence of features → future price/return and optionally uncertainty. Handles variable-length history (e.g. last 30–60 days).
- **Random Forest**
  - Tabular features (indicators + lags) → point prediction and optional prediction intervals (e.g. quantile forests or bootstrap).
- **XGBoost**
  - Same tabular setup; native support for uncertainty (e.g. quantile objective or distribution output) for ranges.
- **Interface**
  - Each model module: `fit(features, targets)`, `predict(features)` returning at least: point forecast, optional lower/upper range, and optionally raw scores for “up” probability.
- **Persistence**
  - Save/load weights or pickled models (paths configurable via `utils` config).

**Outcome:** Three trained models that, given the same feature snapshot, each produce: predicted price (or return), range, and optionally probability of upward movement.

---

### 3.4 `prediction/`

**Purpose:** Orchestrate ensemble and produce the four required outputs per coin.

- **Ensemble**
  - Combine LSTM, RandomForest, and XGBoost outputs (e.g. mean or weighted average for point prediction; min/max or quantile average for ranges).
- **Outputs per coin (14-day horizon)**
  - **Predicted price** — ensemble point forecast (e.g. price at day 14 or path).
  - **Price range** — lower/upper from ensemble ranges (and/or quantiles).
  - **Probability of upward movement** — from classifier head or thresholding returns (e.g. proportion of models voting “up” or average probability).
  - **Confidence score** — derived from model agreement, prediction interval width, or explicit uncertainty head.
- **Orchestration**
  - Load latest features from `features/`, call each model in `models/`, aggregate in `prediction/`, return structured result (e.g. dict or Pydantic model) for the API.

**Outcome:** One clear “run prediction” function that returns, for each of the 10 coins, the four required fields for the next 14 days.

---

### 3.5 `api/`

**Purpose:** Expose predictions and minimal metadata to the dashboard and external clients.

- **Endpoints (suggested)**
  - `GET /coins` — list supported coins (top 10).
  - `GET /predictions` — all coins’ predictions (14-day).
  - `GET /predictions/{symbol}` — single-coin prediction + optional history.
  - `GET /history/{symbol}` — historical prices (and optionally past predictions) for charts.
- **Implementation**
  - FastAPI (or Flask) app; calls into `prediction/` to get current predictions and into `data/` or `features/` for history.
- **Performance**
  - Cache predictions (e.g. TTL 1 hour) so dashboard and users don’t retrain on every request.

**Outcome:** REST API that the dashboard can call to drive selector, tables, and charts.

---

### 3.6 `dashboard/`

**Purpose:** Simple, professional UI for selection, predictions, and charts.

- **Capabilities**
  - **Select cryptocurrency** — dropdown or list of the 10 coins.
  - **View predictions** — table or cards: predicted price, range, P(up), confidence per coin (and for selected coin).
  - **Price charts** — historical price (and optionally prediction overlay) for selected coin; 14-day horizon clearly marked.
- **Tech options**
  - React + Chart.js/Recharts + Tailwind, or Streamlit for a fast, data-focused UI. Static build can call `api/` via configurable base URL.
- **Data flow**
  - All data from API; no direct DB or model access.

**Outcome:** Users can pick a coin, see 14-day predictions and confidence, and inspect history and prediction range on a chart.

---

### 3.7 `utils/`

**Purpose:** Shared code and configuration to keep other modules DRY and consistent.

- **Config**
  - Coin list, date ranges, API keys (via env), paths for data and model artifacts, hyperparameters (or references to config files).
- **Logging**
  - Common logger and (optional) structured logs for training and API.
- **Helpers**
  - Date handling, symbol normalization, safe division for indicators, serialization of predictions.
- **Constants**
  - Symbol-to-id mapping, indicator defaults (e.g. RSI period, EMA spans).

**Outcome:** One place for config and shared utilities; all modules import from here as needed.

---

## 4. Data Flow Summary

1. **data/**  
   Fetches and stores OHLCV + volume; optionally news/social. Exposes reader API for date range and symbols.

2. **features/**  
   Reads from `data/`, computes RSI, MACD, Bollinger Bands, EMA, volatility; builds targets for 14-day horizon and “up” probability. Writes feature matrices and metadata.

3. **models/**  
   Load features and targets from `features/` (or from disk). Train LSTM, RandomForest, XGBoost; save artifacts. Expose `predict(features)` per model.

4. **prediction/**  
   Load latest features; call each model; ensemble → predicted price, range, P(up), confidence per coin; return structured output.

5. **api/**  
   On request, call `prediction/` (with caching) and optionally `data/` or `features/` for history; return JSON.

6. **dashboard/**  
   Call API for list of coins, predictions, and history; render selector, prediction panel, and charts.

---

## 5. Cross-Cutting Concerns

- **Config & secrets:** All keys and environment-specific settings in `utils` (or `config/`); no hardcoding in `data/` or `api/`.
- **Logging:** Centralized in `utils`; each module logs at appropriate level (e.g. debug in feature computation, info in API).
- **Testing:** `tests/` with unit tests per module (e.g. indicator math, ensemble logic, API contract) and optional integration test against a small dataset.
- **Reproducibility:** Fixed train/validation/test splits and seeds; version data and model artifacts if needed (e.g. by date or run id).

---

## 6. Scalability and Production Notes

- **Modularity:** Each folder can be a package; dependencies flow one way (e.g. `prediction` depends on `models` and `features`, not the reverse).
- **Scaling:** Data and feature jobs can be run on a schedule (cron or pipeline); training can be triggered separately; API and dashboard scale independently (stateless).
- **Deployment:** API and dashboard can be containerized (Docker); data and model storage can be local or cloud (S3, RDS) as configured in `utils`.

This structure satisfies the required folders (data, features, models, prediction, api, dashboard, utils), keeps a clear separation of concerns, and supports the 10 coins, 14-day horizon, four output types, multiple data sources, technical indicators, three model types, and ensemble, with a simple professional dashboard and production-friendly layout.
