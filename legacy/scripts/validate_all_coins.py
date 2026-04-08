"""
HTTP smoke + forecast quality table for every supported coin.

Requires a running API (default http://127.0.0.1:8000).

Example::

    python scripts/validate_all_coins.py
    python scripts/validate_all_coins.py --base-url http://127.0.0.1:8080
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.constants import get_supported_coins_list


def _get(url: str, timeout: float = 60.0) -> tuple[int, dict | list | None, str | None]:
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            code = resp.getcode()
            try:
                return code, json.loads(body), None
            except json.JSONDecodeError:
                return code, None, "invalid_json"
    except urllib.error.HTTPError as e:
        try:
            raw = e.read().decode("utf-8")
            return e.code, json.loads(raw), None
        except Exception:
            return e.code, None, str(e)
    except Exception as e:
        return -1, None, str(e)


def main() -> None:
    p = argparse.ArgumentParser(description="Validate forecast/chart/evaluation endpoints per coin.")
    p.add_argument("--base-url", default="http://127.0.0.1:8000", help="API root (no trailing slash)")
    p.add_argument("--timeout", type=float, default=90.0)
    args = p.parse_args()
    base = args.base_url.rstrip("/")

    coins = get_supported_coins_list()
    rows: list[dict] = []

    print(f"{'coin':<12} {'fc':>4} {'ch':>4} {'ev':>4} {'mode':<10} {'valid':<12} {'const':>5} {'deg':>5} {'move14%':>10} {'conf':>6} {'retrain?':>8}")
    print("-" * 100)

    for coin in coins:
        enc = quote(coin, safe="")
        fc_url = f"{base}/api/forecast-path/{enc}"
        ch_url = f"{base}/api/chart/{enc}?days=30"
        ev_url = f"{base}/api/evaluation/{enc}"

        fc_code, fc_data, fc_err = _get(fc_url, timeout=args.timeout)
        ch_code, _, _ = _get(ch_url, timeout=args.timeout)
        ev_code, ev_data, _ = _get(ev_url, timeout=args.timeout)

        row: dict = {
            "coin": coin,
            "forecast_http": fc_code,
            "chart_http": ch_code,
            "evaluation_http": ev_code,
        }
        if isinstance(fc_data, dict) and fc_code == 200:
            cur = float(fc_data.get("current_price") or 0)
            summ = fc_data.get("summary") or {}
            final = float(summ.get("final_day_prediction") or 0)
            move_pct = ((final / cur) - 1.0) * 100.0 if cur > 0 else 0.0
            retrain = (
                fc_data.get("artifact_mode") != "production"
                or fc_data.get("forecast_validity") == "invalid"
                or fc_data.get("is_constant_prediction")
                or not (fc_data.get("forecast_audit") or {}).get("has_evaluation_metrics", True)
            )
            row.update({
                "artifact_mode": fc_data.get("artifact_mode"),
                "forecast_validity": fc_data.get("forecast_validity"),
                "is_constant_prediction": fc_data.get("is_constant_prediction"),
                "degraded_input": fc_data.get("degraded_input"),
                "final_day_move_pct": round(move_pct, 4),
                "confidence_score": fc_data.get("confidence_score"),
                "retrain_recommended": bool(retrain),
            })
            print(
                f"{coin:<12} {fc_code:>4} {ch_code:>4} {ev_code:>4} "
                f"{str(fc_data.get('artifact_mode', '—')):<10} "
                f"{str(fc_data.get('forecast_validity', '—')):<12} "
                f"{str(fc_data.get('is_constant_prediction')):>5} "
                f"{str(fc_data.get('degraded_input')):>5} "
                f"{move_pct:>10.2f} "
                f"{str(fc_data.get('confidence_score', '—')):>6} "
                f"{str(retrain):>8}"
            )
        else:
            row["error"] = fc_err or fc_data
            print(
                f"{coin:<12} {fc_code:>4} {ch_code:>4} {ev_code:>4} "
                f"{'—':<10} {'—':<12} {'—':>5} {'—':>5} {'—':>10} {'—':>6} {'—':>8}"
            )
        rows.append(row)

    out = PROJECT_ROOT / "artifacts" / "validation_summary.json"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
        print(f"\nWrote {out}")
    except OSError as e:
        print(f"\nCould not write summary file: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
