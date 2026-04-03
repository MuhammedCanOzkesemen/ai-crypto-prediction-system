"""
Twitter / X sentiment for crypto (optional multi-source signal).

- Uses VADER for compound sentiment per tweet.
- Fetches via Twitter API v2 when ``TWITTER_BEARER_TOKEN`` is set; otherwise uses
  deterministic mock daily series so training/inference never fails.
- Disk cache limits API calls.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from utils.config import PROJECT_ROOT, settings
from utils.constants import COIN_DISPLAY_TO_ID
from utils.logging_setup import get_logger

logger = get_logger(__name__)

CACHE_DIR = settings.data.artifact_dir / "cache" / "twitter_sentiment"
CACHE_TTL_SEC = float(__import__("os").environ.get("TWITTER_CACHE_TTL_SEC", "21600"))  # 6h
MIN_TWEETS_FOR_CONFIDENCE = 5

_vader_analyzer = None


def _vader():
    global _vader_analyzer
    if _vader_analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer


def clean_text(text: str) -> str:
    """Lowercase, strip URLs, mentions, hashtags, non-letters (keep spaces)."""
    if not text or not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"www\.\S+", " ", t)
    t = re.sub(r"@[\w_]+", " ", t)
    t = re.sub(r"#(\w+)", r"\1", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def compute_sentiment(text: str) -> float:
    """VADER compound score in [-1, 1]."""
    c = clean_text(text)
    if not c:
        return 0.0
    try:
        scores = _vader().polarity_scores(c)
        return float(scores.get("compound", 0.0))
    except Exception as e:
        logger.debug("VADER failed: %s", e)
        return 0.0


def _coin_search_terms(coin_name: str) -> str:
    slug = (coin_name or "").strip()
    cid = COIN_DISPLAY_TO_ID.get(slug, slug.lower().replace(" ", ""))
    sym = {
        "bitcoin": "(bitcoin OR btc)",
        "ethereum": "(ethereum OR eth)",
        "binancecoin": "(bnb OR binance coin)",
        "ripple": "(xrp OR ripple)",
        "solana": "(solana OR sol)",
        "dogecoin": "(dogecoin OR doge)",
        "cardano": "(cardano OR ada)",
        "avalanche-2": "(avalanche OR avax)",
        "chainlink": "(chainlink OR link)",
        "pepe": "(pepe OR pepe coin)",
        "shiba-inu": "(shiba OR shib)",
    }.get(cid, f"({slug.lower()} OR {cid})")
    return f"{sym} lang:en -is:retweet"


def fetch_tweets(coin_name: str, *, max_results: int = 100) -> list[dict[str, Any]]:
    """
    Recent tweets with text + created_at (ISO).

    Uses Twitter API v2 recent search when ``TWITTER_BEARER_TOKEN`` is set;
    otherwise returns mock tweets for the last few days.
    """
    import os

    token = os.environ.get("TWITTER_BEARER_TOKEN", "").strip()
    if not token:
        return _mock_tweets(coin_name, max_results=max_results)

    query = _coin_search_terms(coin_name)
    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "query": query,
        "max_results": min(max(10, max_results), 100),
        "tweet.fields": "created_at,text",
    }
    try:
        r = requests.get(url, headers=headers, params=params, timeout=25)
        if r.status_code != 200:
            logger.warning("Twitter API status %s: %s", r.status_code, r.text[:200])
            return _mock_tweets(coin_name, max_results=max_results)
        data = r.json()
        rows: list[dict[str, Any]] = []
        for t in data.get("data") or []:
            txt = t.get("text") or ""
            ts = t.get("created_at") or ""
            rows.append({"text": txt, "created_at": ts})
        if not rows:
            return _mock_tweets(coin_name, max_results=max_results)
        return rows
    except Exception as e:
        logger.warning("Twitter fetch failed, using mock: %s", e)
        return _mock_tweets(coin_name, max_results=max_results)


def _mock_tweets(coin_name: str, *, max_results: int) -> list[dict[str, Any]]:
    seed = int(hashlib.sha256((coin_name or "btc").encode()).hexdigest()[:8], 16) % (2**31 - 1)
    rng = np.random.RandomState(seed)
    n = min(max(12, max_results // 2), max_results)
    templates = [
        "{} looking strong today momentum building",
        "bearish {} here careful",
        "{} pump incoming",
        "not sure about {} short term",
        "{} consolidation boring",
        "love the {} chart",
        "{} crash fear",
        "holding {}",
    ]
    out: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc)
    for i in range(n):
        tmpl = templates[int(rng.randint(0, len(templates)))]
        sym = (coin_name or "crypto").split()[0]
        phrase = tmpl.format(sym.lower())
        if rng.rand() > 0.5:
            phrase += " https://t.co/xxxxx"
        if rng.rand() > 0.4:
            phrase += " #" + sym[:4]
        dt = now - timedelta(hours=int(rng.randint(0, 168)))
        out.append({"text": phrase, "created_at": dt.isoformat().replace("+00:00", "Z")})
    return out


def _cache_path(coin_name: str) -> Path:
    slug = (coin_name or "unknown").replace(" ", "_")
    return CACHE_DIR / f"{slug}_recent_tweets.json"


def _load_cache(coin_name: str) -> list[dict[str, Any]] | None:
    p = _cache_path(coin_name)
    if not p.exists():
        return None
    try:
        age = time.time() - p.stat().st_mtime
        if age > CACHE_TTL_SEC:
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        rows = data.get("tweets")
        return rows if isinstance(rows, list) else None
    except Exception:
        return None


def _save_cache(coin_name: str, tweets: list[dict[str, Any]]) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        p = _cache_path(coin_name)
        p.write_text(
            json.dumps({"fetched_at": time.time(), "tweets": tweets}, indent=0),
            encoding="utf-8",
        )
    except OSError as e:
        logger.debug("Twitter cache write failed: %s", e)


def get_twitter_sentiment_series(coin_name: str) -> pd.DataFrame:
    """
    Daily aggregates aligned by calendar date (UTC).

    Columns:
      - sentiment_score: mean VADER compound
      - sentiment_volume: tweet count
      - sentiment_momentum_3d: rolling 3-day change in sentiment_score

    Empty → single row of zeros for today, or empty frame handled upstream.
    """
    cached = _load_cache(coin_name)
    tweets = cached if cached is not None else fetch_tweets(coin_name, max_results=100)
    if cached is None:
        _save_cache(coin_name, tweets)

    if not tweets:
        today = datetime.now(timezone.utc).date()
        return pd.DataFrame(
            {
                "date": [pd.Timestamp(today)],
                "sentiment_score": [0.0],
                "sentiment_volume": [0.0],
                "sentiment_momentum_3d": [0.0],
            }
        )

    rows: list[tuple[date, float]] = []
    for tw in tweets:
        txt = tw.get("text", "")
        ts = tw.get("created_at", "")
        try:
            dtp = pd.to_datetime(ts, utc=True)
            d = dtp.date()
        except Exception:
            d = datetime.now(timezone.utc).date()
        rows.append((d, compute_sentiment(txt)))

    df = pd.DataFrame(rows, columns=["day", "compound"])
    g = df.groupby("day", as_index=False).agg(sentiment_score=("compound", "mean"), sentiment_volume=("compound", "count"))
    g = g.sort_values("day").reset_index(drop=True)
    g["date"] = pd.to_datetime(g["day"]).dt.normalize()
    g = g.drop(columns=["day"])
    g["sentiment_momentum_3d"] = g["sentiment_score"].diff(3).fillna(0.0)
    g["twitter_sentiment_7d_avg"] = g["sentiment_score"].rolling(7, min_periods=1).mean().fillna(0.0)
    g["twitter_sentiment_momentum"] = g["sentiment_momentum_3d"]
    return g[["date", "sentiment_score", "sentiment_volume", "sentiment_momentum_3d", "twitter_sentiment_7d_avg", "twitter_sentiment_momentum"]]


def daily_sentiment_features_for_price_dates(
    coin_name: str | None,
    price_dates: pd.Series,
) -> pd.DataFrame:
    """
    Build a daily sentiment table covering ``price_dates`` min..max.

    Merges API/mock recent window with deterministic mock fill for older dates
    so long training histories always get finite features (neutral-ish baseline).
    """
    if coin_name is None or not str(coin_name).strip():
        return pd.DataFrame()

    dmin = pd.to_datetime(price_dates.min()).normalize()
    dmax = pd.to_datetime(price_dates.max()).normalize()
    idx = pd.date_range(dmin, dmax, freq="D")

    seed = int(hashlib.sha256(f"{coin_name}|twitter_mock".encode()).hexdigest()[:12], 16) % (2**31 - 1)
    rng = np.random.RandomState(seed)
    base = pd.DataFrame({"date": idx})
    base["sentiment_score"] = (rng.randn(len(base)) * 0.08).clip(-0.35, 0.35)
    base["sentiment_volume"] = rng.poisson(22, size=len(base)).astype(float)
    base.loc[base["sentiment_volume"] < 0, "sentiment_volume"] = 0.0

    try:
        recent = get_twitter_sentiment_series(coin_name)
        if recent is not None and not recent.empty:
            recent = recent.copy()
            recent["date"] = pd.to_datetime(recent["date"]).dt.tz_localize(None).dt.normalize()
            r = recent.set_index("date")[["sentiment_score", "sentiment_volume"]]
            b = base.set_index("date")
            b.update(r)
            base = b.reset_index()
    except Exception as e:
        logger.debug("Twitter overlay skipped: %s", e)

    base["sentiment_score"] = base["sentiment_score"].fillna(0.0).clip(-1.0, 1.0)
    base["sentiment_volume"] = base["sentiment_volume"].fillna(0.0).clip(0.0, 1e6)
    base["sentiment_momentum_3d"] = base["sentiment_score"].diff(3).fillna(0.0)
    base["twitter_sentiment_7d_avg"] = base["sentiment_score"].rolling(7, min_periods=1).mean().fillna(0.0)
    base["twitter_sentiment_momentum"] = base["sentiment_momentum_3d"]
    return base
