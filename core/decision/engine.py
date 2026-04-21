from typing import Optional


def decide(
    probability: float,
    realized_vol_5d: float,
    last_trade_day: Optional[int],
    current_day: int,
    probability_threshold: float = 0.58,
) -> dict:
    reasons = []

    if probability < probability_threshold:
        reasons.append("probability_below_threshold")

    if not (0.005 < realized_vol_5d < 0.08):
        reasons.append("volatility_out_of_range")

    if last_trade_day is not None and current_day - last_trade_day < 1:
        reasons.append("cooldown_active")

    signal = "NO_TRADE" if reasons else "BUY"

    return {
        "signal": signal,
        "probability": probability,
        "reasons": reasons,
    }
