"""
Shared constants and coin mappings for the cryptocurrency forecasting platform.

Single source of truth for supported coins and standard data schemas.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Supported coins: display name -> CoinGecko API id
# Order reflects priority / display order.
# -----------------------------------------------------------------------------

COIN_DISPLAY_TO_ID: dict[str, str] = {
    "Bitcoin": "bitcoin",
    "Ethereum": "ethereum",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "Solana": "solana",
    "Dogecoin": "dogecoin",
    "Cardano": "cardano",
    "Avalanche": "avalanche-2",
    "Chainlink": "chainlink",
    "PEPE": "pepe",
    "SHIB": "shiba-inu",
}

# Canonical list of supported coin display names (for API and pipelines)
SUPPORTED_COINS: tuple[str, ...] = tuple(COIN_DISPLAY_TO_ID.keys())

# Binance spot symbol mapping (display name -> symbol)
COIN_DISPLAY_TO_BINANCE: dict[str, str] = {
    "Bitcoin": "BTCUSDT",
    "Ethereum": "ETHUSDT",
    "BNB": "BNBUSDT",
    "XRP": "XRPUSDT",
    "Solana": "SOLUSDT",
    "Dogecoin": "DOGEUSDT",
    "Cardano": "ADAUSDT",
    "Avalanche": "AVAXUSDT",
    "Chainlink": "LINKUSDT",
    "PEPE": "PEPEUSDT",
    "SHIB": "SHIBUSDT",
}


def get_supported_coins_list() -> list[str]:
    """Return supported coin display names as a list."""
    return list(SUPPORTED_COINS)


# -----------------------------------------------------------------------------
# Data schema
# -----------------------------------------------------------------------------

PRICE_COLUMNS: tuple[str, ...] = ("date", "open", "high", "low", "close", "volume")

# -----------------------------------------------------------------------------
# Feature / indicator defaults (can be overridden via config)
# -----------------------------------------------------------------------------

RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_NUM_STD = 2.0
EMA_SHORT = 20
EMA_LONG = 50
SMA_SHORT = 20
SMA_LONG = 50
ATR_PERIOD = 14
VOLATILITY_ROLLING = 14
TARGET_HORIZON_DAYS = 14

# Lag periods for lag features
LAG_PERIODS = (1, 2, 3, 5, 7)

# ADX / ROC / volatility regime (log-return EWMA scale)
ADX_PERIOD = 14
ROC_PERIODS = (5, 10)
VOL_REGIME_LOGBOUND_LOW = 0.012
VOL_REGIME_LOGBOUND_HIGH = 0.030

# Data fetching: delay between coins to stay under CoinGecko free tier (~30/min)
COIN_FETCH_INTERVAL_SEC = 4
