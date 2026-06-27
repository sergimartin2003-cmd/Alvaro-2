"""Análisis de la estructura temporal de VIX (contango / backwardation)."""

from __future__ import annotations


class VIXTermStructureAnalyzer:
    """Calcula pendiente de la curva VIX y genera señales de volatilidad."""

    def calculate_term_structure(self, vix_futures_prices: dict) -> dict | None:
        m1 = vix_futures_prices.get(30)
        m2 = vix_futures_prices.get(60)
        if m1 is None or m2 is None:
            return None
        slope = (m2 - m1) / 30
        roll_yield = (m2 - m1) / m1 if m1 else 0.0
        return {
            "m1": m1,
            "m2": m2,
            "slope": slope,
            "roll_yield": roll_yield,
            "regime": "CONTANGO" if m2 > m1 else "BACKWARDATION",
        }

    def generate_signal(self, term_structure: dict):
        if term_structure is None:
            return "HOLD", "Datos insuficientes"
        if term_structure["regime"] == "BACKWARDATION":
            if term_structure["m1"] > 30:
                return "BUY_EQUITIES", "Miedo extremo - señal contrarian de compra"
            return "BUY_VOLATILITY", "Backwardation - spike de volatilidad"
        if term_structure["roll_yield"] > 0.05:
            return "SELL_VOLATILITY", "Contango fuerte - vender volatilidad"
        return "HOLD", "Contango normal"
