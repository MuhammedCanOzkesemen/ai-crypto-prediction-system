from core.decision.engine import decide


def test_low_probability_no_trade():
    result = decide(probability=0.50, realized_vol_5d=0.02, last_trade_day=None, current_day=10)
    assert result["signal"] == "NO_TRADE"
    assert "probability_below_threshold" in result["reasons"]


def test_high_volatility_no_trade():
    result = decide(probability=0.65, realized_vol_5d=0.08, last_trade_day=None, current_day=10)
    assert result["signal"] == "NO_TRADE"
    assert "volatility_too_high" in result["reasons"]


def test_cooldown_no_trade():
    result = decide(probability=0.65, realized_vol_5d=0.02, last_trade_day=10, current_day=10)
    assert result["signal"] == "NO_TRADE"
    assert "cooldown_active" in result["reasons"]


def test_valid_case_buy():
    result = decide(probability=0.65, realized_vol_5d=0.02, last_trade_day=9, current_day=10)
    assert result["signal"] == "BUY"
    assert result["reasons"] == []
    assert result["probability"] == 0.65


def test_multiple_reasons():
    result = decide(probability=0.40, realized_vol_5d=0.09, last_trade_day=10, current_day=10)
    assert result["signal"] == "NO_TRADE"
    assert len(result["reasons"]) == 3
