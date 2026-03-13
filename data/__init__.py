"""
Data module: ingestion, storage, and schemas for cryptocurrency price data.

Uses CoinGecko API; see price_fetcher for OHLCV fetching and CSV export.
"""

from .price_fetcher import (
    STANDARD_COLUMNS,
    fetch_historical_data,
    get_coin_list,
    get_supported_coins,
    save_data_to_csv,
)

__all__ = [
    "STANDARD_COLUMNS",
    "fetch_historical_data",
    "get_coin_list",
    "get_supported_coins",
    "save_data_to_csv",
]
