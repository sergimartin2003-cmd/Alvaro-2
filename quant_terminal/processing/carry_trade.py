"""Estrategias de carry trade basadas en diferenciales de tasas."""

from __future__ import annotations


class CarryTradeAnalyzer:
    """Oportunidades de carry y evaluación de riesgo por régimen de mercado."""

    def __init__(self, interest_rates: dict | None = None) -> None:
        self.interest_rates = interest_rates or {
            "USD": 5.50, "EUR": 4.50, "GBP": 5.25, "JPY": 0.10, "CHF": 1.75,
            "AUD": 4.35, "NZD": 5.50, "CAD": 5.00, "MXN": 11.25, "BRL": 13.75,
            "ZAR": 8.25, "TRY": 50.00,
        }

    def calculate_carry_opportunities(self, volatility_data: dict | None = None) -> list[dict]:
        opps = []
        currencies = list(self.interest_rates)
        for long_c in currencies:
            for short_c in currencies:
                if long_c == short_c:
                    continue
                carry = self.interest_rates[long_c] - self.interest_rates[short_c]
                if carry <= 3.0:
                    continue
                if volatility_data:
                    vol = volatility_data.get(f"{long_c}{short_c}", 10.0)
                    risk_adj = carry / vol if vol else carry
                else:
                    risk_adj = carry
                opps.append(
                    {
                        "pair": f"{long_c}/{short_c}",
                        "long": long_c,
                        "short": short_c,
                        "annual_carry": carry,
                        "risk_adjusted_carry": risk_adj,
                        "direction": "LONG",
                    }
                )
        opps.sort(key=lambda x: x["risk_adjusted_carry"], reverse=True)
        return opps

    def evaluate_risk(self, carry_trade: dict, market_regime: dict):
        score = 0
        if market_regime.get("volatility") == "HIGH":
            score += 3
        if market_regime.get("risk_appetite") == "RISK-OFF":
            score += 2
        if market_regime.get("geopolitical_tension") == "HIGH":
            score += 2
        if score > 5:
            return "AVOID", "Entorno de alto riesgo para carry trades"
        if score > 3:
            return "CAUTION", "Riesgo moderado - reducir tamaño"
        return "FAVORABLE", "Buen entorno para carry trades"
