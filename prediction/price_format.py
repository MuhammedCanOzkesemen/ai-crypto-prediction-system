"""
Dynamic decimal precision for very small asset prices (e.g. PEPE).

Fixed 6-decimal rounding collapses sub-cent prices to a flat line in JSON/API.
"""

from __future__ import annotations

import math


def price_decimal_places(reference_price: float, *, min_dp: int = 6, max_dp: int = 14) -> int:
    """
    Decimal places so typical daily moves remain visible vs magnitude of reference price.

    Examples: BTC ~68k → 6 dp; PEPE ~3e-6 → ~11–12 dp.
    """
    if not math.isfinite(reference_price) or reference_price <= 0:
        return max_dp
    le = math.log10(reference_price)
    extra = int(math.ceil(-le)) + 5
    return max(min_dp, min(max_dp, extra))


def round_price(x: float, reference_price: float) -> float:
    d = price_decimal_places(reference_price)
    return round(float(x), d)
