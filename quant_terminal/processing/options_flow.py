"""Análisis de flujo de opciones y dark pools (smart money tracking)."""

from __future__ import annotations

from collections import defaultdict


class OptionsFlowAnalyzer:
    """Detecta actividad inusual, sweeps, dark pool prints y Put/Call ratio."""

    def detect_unusual_activity(self, options_chain: list[dict]) -> list[dict]:
        signals = []
        for o in options_chain:
            vol = o.get("volume", 0)
            oi = max(o.get("open_interest", 0), 1)
            ratio = vol / oi
            direction = "CALL" if o.get("type", "C").upper().startswith("C") else "PUT"
            if ratio > 5:
                signals.append(
                    {
                        "type": "UNUSUAL_VOLUME",
                        "symbol": o.get("symbol"),
                        "strike": o.get("strike"),
                        "expiry": o.get("expiry"),
                        "volume": vol,
                        "oi": oi,
                        "ratio": ratio,
                        "direction": direction,
                    }
                )
            if o.get("is_sweep"):
                signals.append(
                    {
                        "type": "SWEEP",
                        "symbol": o.get("symbol"),
                        "notional": o.get("premium", 0) * vol * 100,
                        "direction": direction,
                    }
                )
        return signals

    def analyze_dark_pool_activity(self, dark_pool_trades: list[dict]) -> list[dict]:
        significant = [
            t for t in dark_pool_trades
            if t.get("size", 0) > t.get("avg_daily_volume", 0) * 0.1
        ]
        by_ticker: dict[str, list] = defaultdict(list)
        for t in significant:
            by_ticker[t["symbol"]].append(t)

        out = []
        for ticker, trades in by_ticker.items():
            total = sum(t["size"] for t in trades)
            avg_price = sum(t["price"] * t["size"] for t in trades) / total
            out.append(
                {
                    "symbol": ticker,
                    "dark_pool_volume": total,
                    "avg_price": avg_price,
                    "num_trades": len(trades),
                    "sentiment": "BULLISH"
                    if avg_price > trades[0].get("close", avg_price)
                    else "BEARISH",
                }
            )
        return out

    def calculate_put_call_ratio(self, options_data: list[dict]) -> dict:
        put = sum(o["volume"] for o in options_data if o.get("type", "").upper().startswith("P"))
        call = sum(o["volume"] for o in options_data if o.get("type", "").upper().startswith("C"))
        pcr = put / max(call, 1)
        sentiment = "BULLISH" if pcr > 1.2 else "BEARISH" if pcr < 0.7 else "NEUTRAL"
        return {"put_call_ratio": pcr, "sentiment": sentiment}
