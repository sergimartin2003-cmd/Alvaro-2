"""Detección de arbitraje de latencia (PROMPT 3.9).

Quotes obsoletos, arbitraje cross-venue y medición de la ventaja de latencia.
numpy/pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class LatencyArbitrageDetector:
    def detect_stale_quotes(self, quotes: pd.DataFrame, max_latency_ms: int = 100) -> list[dict]:
        """Quotes cuya antigüedad supera el umbral => potencialmente explotables."""
        out = []
        for ts, row in quotes.iterrows():
            age = row.get("age_ms", 0)
            if age > max_latency_ms:
                out.append({
                    "timestamp": ts,
                    "age_ms": int(age),
                    "arbitrage_opportunity": True,
                    "expected_profit": float(row.get("expected_move", 0.0)),
                })
        return out

    def detect_cross_venue_arbitrage(self, quotes_by_venue: dict) -> list[dict]:
        """Si bid(v1) > ask(v2): comprar en v2, vender en v1."""
        out = []
        venues = list(quotes_by_venue)
        for i, v1 in enumerate(venues):
            for v2 in venues:
                if v1 == v2:
                    continue
                bid1 = quotes_by_venue[v1].get("bid", 0)
                ask2 = quotes_by_venue[v2].get("ask", np.inf)
                if bid1 > ask2:
                    out.append({
                        "buy_venue": v2,
                        "sell_venue": v1,
                        "expected_profit": float(bid1 - ask2),
                        "required_latency": float(
                            min(quotes_by_venue[v1].get("latency_ms", 1),
                                quotes_by_venue[v2].get("latency_ms", 1))
                        ),
                    })
        out.sort(key=lambda x: x["expected_profit"], reverse=True)
        return out

    def measure_latency_advantage(self, your_latency_ms: float, market_latency_ms: float) -> dict:
        adv = market_latency_ms - your_latency_ms
        if adv <= 0:
            level = "NONE"
        elif adv < 1:
            level = "MARGINAL"
        elif adv < 10:
            level = "SIGNIFICANT"
        else:
            level = "DOMINANT"
        return {
            "latency_advantage_ms": float(adv),
            "advantage_level": level,
            "exploitable": adv > 0,
            "expected_profit_per_opportunity": float(max(adv, 0) * 0.001),
        }
