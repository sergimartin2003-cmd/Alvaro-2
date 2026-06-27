"""Estrategias de carry trade basadas en diferenciales de tasas."""

from __future__ import annotations

import numpy as np


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


class CarryTradeOptimizer:
    """Optimización de carry (PROMPT 1.13): oportunidades, portfolio y monitoreo."""

    def calculate_carry_opportunities(self, interest_rates: dict, fx_volatility: dict | None = None) -> list[dict]:
        out = []
        currencies = list(interest_rates)
        for long_c in currencies:
            for short_c in currencies:
                if long_c == short_c:
                    continue
                carry = interest_rates[long_c] - interest_rates[short_c]
                if carry <= 0:
                    continue
                pair = f"{long_c}{short_c}"
                vol = (fx_volatility or {}).get(pair, 10.0)
                risk_adj = carry / vol if vol else carry
                out.append({
                    "pair": f"{long_c}/{short_c}",
                    "long_currency": long_c,
                    "short_currency": short_c,
                    "annual_carry": float(carry),
                    "volatility": float(vol),
                    "risk_adjusted_carry": float(risk_adj),
                    "sharpe_ratio": float(risk_adj),
                })
        out.sort(key=lambda x: x["risk_adjusted_carry"], reverse=True)
        return out

    def optimize_carry_portfolio(self, opportunities: list[dict], max_positions: int = 5,
                                 max_weight: float = 0.20) -> dict:
        selected = opportunities[:max_positions]
        if not selected:
            return {"portfolio": {}, "weights": {}, "expected_carry": 0.0,
                    "expected_volatility": 0.0, "sharpe_ratio": 0.0}
        scores = np.array([o["risk_adjusted_carry"] for o in selected])
        w = np.minimum(scores / scores.sum(), max_weight)
        w = w / w.sum()
        weights = {o["pair"]: float(wi) for o, wi in zip(selected, w)}
        exp_carry = float(sum(wi * o["annual_carry"] for wi, o in zip(w, selected)))
        exp_vol = float(np.sqrt(sum((wi * o["volatility"]) ** 2 for wi, o in zip(w, selected))))
        return {
            "portfolio": selected,
            "weights": weights,
            "expected_carry": exp_carry,
            "expected_volatility": exp_vol,
            "sharpe_ratio": float(exp_carry / exp_vol) if exp_vol else 0.0,
        }

    def monitor_carry_trades(self, portfolio: list[dict], pnl: dict | None = None) -> list[dict]:
        pnl = pnl or {}
        out = []
        for pos in portfolio:
            p = pnl.get(pos["pair"], 0.0)
            out.append({
                "position": pos["pair"],
                "pnl": float(p),
                "carry_accumulated": float(pos["annual_carry"] / 252),
                "exit_signal": p < -2 * pos["volatility"] / 100,
                "risk_warning": "unwind risk" if pos["volatility"] > 15 else "",
            })
        return out
