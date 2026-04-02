"""
Data module: ingestion, storage, and schemas for cryptocurrency price data.

Uses CoinGecko API; see price_fetcher for OHLCV fetching and CSV export.
"""

from .data_refresh import (
    DataFreshness,
    check_data_freshness,
    get_last_refresh_time,
    refresh_data_if_needed,
)
from .price_fetcher import (
    STANDARD_COLUMNS,
    fetch_historical_data,
    get_coin_list,
    get_supported_coins,
    save_data_to_csv,
)

__all__ = [
    "STANDARD_COLUMNS",
    "DataFreshness",
    "check_data_freshness",
    "fetch_historical_data",
    "get_coin_list",
    "get_last_refresh_time",
    "get_supported_coins",
    "refresh_data_if_needed",
    "save_data_to_csv",
]
