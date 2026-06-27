"""Smart Order Router (PROMPT 3.8).

Selección de venue, división de órdenes entre venues y monitoreo de la calidad
de ejecución. numpy.
"""

from __future__ import annotations

import numpy as np


class SmartOrderRouter:
    def _venue_score(self, venue: dict, side: str) -> float:
        """Mayor score = mejor. Prioriza precio, liquidez, fees y latencia."""
        price = venue.get("ask" if side == "BUY" else "bid", 0)
        price_term = -price if side == "BUY" else price
        liquidity = venue.get("liquidity", 0)
        fee = venue.get("fee_bps", 0)
        latency = venue.get("latency_ms", 0)
        return price_term * 100 + liquidity / 1000 - fee - latency / 10

    def select_best_venue(self, order: dict, venues: dict) -> dict:
        side = order.get("side", "BUY")
        scores = {name: self._venue_score(v, side) for name, v in venues.items()}
        best = max(scores, key=scores.get)
        return {
            "best_venue": best,
            "venue_scores": scores,
            "routing_decision": f"Enrutar a {best}",
        }

    def split_order_across_venues(self, total_quantity: float, venues: dict) -> dict:
        """Reparte proporcional a la liquidez disponible en cada venue."""
        liq = {n: v.get("liquidity", 0) for n, v in venues.items()}
        total_liq = sum(liq.values()) or 1.0
        allocations = {n: float(total_quantity * liq[n] / total_liq) for n in venues}
        expected_cost = sum(
            allocations[n] * venues[n].get("fee_bps", 0) / 10000 for n in venues
        )
        return {
            "venue_allocations": allocations,
            "total_expected_cost": float(expected_cost),
            "execution_plan": [{"venue": n, "qty": q} for n, q in allocations.items()],
        }

    def monitor_execution_quality(self, fills: list[dict], benchmark_price: float) -> dict:
        if not fills:
            return {"avg_slippage_bps": 0.0, "venue_performance": {}, "recommendations": []}
        slippages = []
        by_venue: dict[str, list] = {}
        for f in fills:
            slip = (f["price"] - benchmark_price) / benchmark_price * 10000
            slippages.append(slip)
            by_venue.setdefault(f.get("venue", "?"), []).append(slip)
        venue_perf = {v: float(np.mean(s)) for v, s in by_venue.items()}
        worst = max(venue_perf, key=venue_perf.get) if venue_perf else None
        recs = [f"Revisar enrutamiento a {worst}"] if worst and venue_perf[worst] > 5 else []
        return {
            "avg_slippage_bps": float(np.mean(slippages)),
            "venue_performance": venue_perf,
            "recommendations": recs,
        }
