"""Gestión de riesgo: position sizing, VaR y evaluación de riesgo por trade."""

from __future__ import annotations

import numpy as np


class RiskManager:
    """Dimensiona posiciones y controla la exposición total del portfolio."""

    def __init__(self, portfolio_value: float, max_risk_per_trade: float = 0.02,
                 max_total_exposure: float = 0.20) -> None:
        self.portfolio_value = portfolio_value
        self.max_risk_per_trade = max_risk_per_trade
        self.max_total_exposure = max_total_exposure
        self.current_positions: dict[str, dict] = {}
        self.portfolio_volatility = 0.015  # estimación diaria por defecto

    def calculate_position_size(self, signal: dict, asset_price: float,
                                stop_loss_price: float) -> dict:
        risk_per_share = abs(asset_price - stop_loss_price)
        if risk_per_share <= 0:
            risk_per_share = asset_price * 0.01
        max_risk = self.portfolio_value * self.max_risk_per_trade
        shares = int(max_risk / risk_per_share)
        dollars = shares * asset_price

        current_exposure = sum(p["risk_amount"] for p in self.current_positions.values())
        approved = (current_exposure + max_risk) <= (self.portfolio_value * self.max_total_exposure)

        return {
            "position_size_shares": shares,
            "position_size_dollars": dollars,
            "risk_amount": max_risk,
            "risk_per_share": risk_per_share,
            "risk_reward_ratio": signal.get("risk_reward_ratio", 0.0),
            "position_as_percent_of_portfolio": dollars / self.portfolio_value,
            "approved": approved,
            "reasoning": "Posición aprobada" if approved else "Supera el límite de exposición total",
        }

    def calculate_portfolio_var(self, confidence: float = 0.95, horizon_days: int = 1) -> dict:
        from scipy.stats import norm

        vol = self.portfolio_volatility
        z = abs(norm.ppf(1 - confidence))
        var_95 = z * vol * self.portfolio_value * np.sqrt(horizon_days)
        z99 = abs(norm.ppf(0.99))
        var_99 = z99 * vol * self.portfolio_value * np.sqrt(horizon_days)
        es_95 = (norm.pdf(norm.ppf(1 - confidence)) / (1 - confidence)) * vol * self.portfolio_value * np.sqrt(horizon_days)
        return {
            "var_95": var_95,
            "var_99": var_99,
            "expected_shortfall_95": es_95,
            "portfolio_volatility": vol,
            "var_as_percent_of_portfolio": var_95 / self.portfolio_value,
        }

    def assess_trade_risk(self, signal: dict, position_size: dict) -> dict:
        score = 0.0
        factors, recs = [], []

        if position_size["position_as_percent_of_portfolio"] > 0.15:
            score += 0.15
            factors.append("Tamaño de posición grande")
            recs.append("Reducir el tamaño de la posición")

        if position_size.get("risk_reward_ratio", 0) < 2.0:
            score += 0.15
            factors.append("Ratio riesgo/beneficio pobre")
            recs.append("Mejorar el R/R o descartar el trade")

        if not position_size.get("approved", True):
            score += 0.4
            factors.append("Excede el límite de exposición total")

        if signal.get("confidence", 1.0) < 0.5:
            score += 0.2
            factors.append("Baja confianza en la señal")

        if score < 0.3:
            level = "LOW"
        elif score < 0.6:
            level = "MEDIUM"
        elif score < 0.8:
            level = "HIGH"
        else:
            level = "EXTREME"

        return {
            "risk_score": score,
            "risk_level": level,
            "risk_factors": factors,
            "approved": score < 0.7,
            "recommendations": recs,
        }
