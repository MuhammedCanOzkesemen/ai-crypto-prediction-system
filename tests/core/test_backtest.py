import numpy as np
import pandas as pd
import pytest

from core.backtest.simulator import simulate


@pytest.fixture(scope="module")
def synthetic_data():
    rng = np.random.default_rng(0)
    n = 320
    index = pd.date_range("2020-01-01", periods=n, freq="D")

    features = pd.DataFrame(rng.standard_normal((n, 9)), index=index)
    features.columns = [f"feature_{i}" for i in range(9)]
    features["realized_vol_5d"] = rng.uniform(0.01, 0.04, size=n)

    labels = pd.Series(rng.integers(0, 2, size=n), index=index)

    price = np.cumprod(1 + rng.normal(0.0005, 0.01, size=n)) * 30000
    spread = rng.uniform(0.002, 0.012, size=n)
    ohlc = pd.DataFrame({
        "open": price * (1 + rng.uniform(-0.003, 0.003, size=n)),
        "high": price * (1 + spread),
        "low": price * (1 - spread),
        "close": price * (1 + rng.uniform(-0.003, 0.003, size=n)),
    }, index=index)

    return features, labels, ohlc


@pytest.fixture(scope="module")
def trades(synthetic_data):
    features, labels, ohlc = synthetic_data
    return simulate(features, labels, ohlc, min_history=200)


def test_trades_not_empty(trades):
    assert len(trades) >= 1


def test_returns_are_finite(trades):
    for trade in trades:
        assert np.isfinite(trade["return"])


def test_trade_has_required_keys(trades):
    for trade in trades:
        assert "entry_price" in trade
        assert "exit_price" in trade
        assert "return" in trade
