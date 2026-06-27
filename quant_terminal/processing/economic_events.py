"""Análisis de publicaciones de eventos económicos y expectativas de la Fed."""

from __future__ import annotations


class EconomicEventAnalyzer:
    """Surprise factor, impacto estimado y expectativas de tipos."""

    def __init__(self) -> None:
        self.event_importance = {
            "Non-Farm Payrolls": {"impact": 0.95, "assets": ["USD", "SPY", "TLT"]},
            "CPI": {"impact": 0.90, "assets": ["USD", "TLT", "GLD"]},
            "FOMC Rate Decision": {"impact": 0.98, "assets": ["USD", "SPY", "TLT", "GLD"]},
            "GDP": {"impact": 0.85, "assets": ["USD", "SPY"]},
            "Retail Sales": {"impact": 0.75, "assets": ["USD", "XRT"]},
            "PMI": {"impact": 0.70, "assets": ["USD", "SPY"]},
        }

    def analyze_event_release(self, event: dict) -> dict:
        actual = event.get("actual")
        forecast = event.get("forecast")
        previous = event.get("previous")
        std = event.get("std_dev") or (abs(forecast) * 0.1 if forecast else 1.0) or 1.0

        surprise = (actual - forecast) / std if actual is not None and forecast is not None else 0.0
        direction = "POSITIVE" if surprise > 0 else "NEGATIVE" if surprise < 0 else "NEUTRAL"
        magnitude = "STRONG" if abs(surprise) > 2 else "MODERATE" if abs(surprise) > 1 else "WEAK"

        name = event.get("event", "")
        info = self.event_importance.get(name, {"impact": 0.5, "assets": ["SPY"]})

        # Heurística simple: sorpresa positiva en datos de actividad → USD/SPY arriba, bonos abajo.
        impact = {}
        for asset in info["assets"]:
            if asset in ("USD", "SPY", "QQQ"):
                impact[asset] = {"direction": direction, "magnitude": 0.01 * abs(surprise)}
            elif asset in ("TLT", "IEF", "GLD"):
                opp = "NEGATIVE" if direction == "POSITIVE" else "POSITIVE"
                impact[asset] = {"direction": opp, "magnitude": 0.01 * abs(surprise)}

        return {
            "surprise_analysis": {
                "surprise_factor": surprise,
                "surprise_direction": direction,
                "surprise_magnitude": magnitude,
                "revision": previous,
            },
            "event_importance": info["impact"],
            "market_impact_prediction": impact,
        }

    def calculate_fed_expectations(self, fed_funds_futures: dict, current_rate: float = 5.50) -> dict:
        path = []
        for meeting, price in sorted(fed_funds_futures.items()):
            implied = 100 - price
            path.append({"date": meeting, "expected_rate": round(implied, 3)})
        outlook = "STABLE"
        if path:
            if path[-1]["expected_rate"] < current_rate - 0.1:
                outlook = "DOVISH (cuts priced in)"
            elif path[-1]["expected_rate"] > current_rate + 0.1:
                outlook = "HAWKISH (hikes priced in)"
        return {"current_rate": current_rate, "policy_path": path, "outlook": outlook}
