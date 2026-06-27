"""Análisis de política monetaria (CME FedWatch) a partir de fed funds futures."""

from __future__ import annotations


class FedWatchAnalyzer:
    """Probabilidades de hike/cut/hold y señales de trading asociadas."""

    def __init__(self, current_rate: float = 5.50) -> None:
        self.current_rate = current_rate

    def calculate_rate_probabilities(self, fed_funds_futures: dict) -> dict:
        probs = {}
        for meeting, price in fed_funds_futures.items():
            implied = 100 - price
            diff = implied - self.current_rate
            if diff < -0.125:
                p_cut = min(abs(diff) / 0.25, 1.0)
                p_hike, p_hold = 0.0, 1 - p_cut
            elif diff > 0.125:
                p_hike = min(abs(diff) / 0.25, 1.0)
                p_cut, p_hold = 0.0, 1 - p_hike
            else:
                p_hold = 1 - abs(diff) / 0.125
                p_cut = p_hike = (1 - p_hold) / 2
            probs[meeting] = {
                "implied_rate": implied,
                "prob_cut": p_cut,
                "prob_hold": p_hold,
                "prob_hike": p_hike,
                "expected_move": diff,
            }
        return probs

    def generate_trading_signals(self, rate_probabilities: dict, economic_data: dict) -> dict:
        if not rate_probabilities:
            return {}
        next_meeting = min(rate_probabilities)
        probs = rate_probabilities[next_meeting]
        strength = economic_data.get("strength", "NEUTRAL")
        signals: dict[str, str] = {}
        if probs["prob_cut"] > 0.7:
            if strength == "STRONG":
                signals = {"USD": "BUY", "BONDS": "SELL", "EQUITIES": "NEUTRAL"}
            else:
                signals = {"USD": "SELL", "BONDS": "BUY", "EQUITIES": "BUY"}
        elif probs["prob_hike"] > 0.7:
            if strength == "WEAK":
                signals = {"USD": "SELL", "BONDS": "BUY", "EQUITIES": "BUY"}
            else:
                signals = {"USD": "BUY", "BONDS": "SELL", "EQUITIES": "SELL"}
        return signals
