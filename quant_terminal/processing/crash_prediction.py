"""Predicción de crashes (PROMPT 1.12).

Critical slowing down (varianza/autocorrelación crecientes), señales de alerta
temprana y simulación de escenarios de crash sobre un portfolio. numpy/pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class CrashPredictionEngine:
    def detect_critical_slowing_down(self, series: pd.Series, window: int = 60) -> dict:
        s = pd.Series(series).dropna()
        var = s.rolling(20).var().dropna()
        ac1 = s.rolling(20).apply(lambda x: pd.Series(x).autocorr(lag=1), raw=False).dropna()

        def _trend(x):
            if len(x) < 5:
                return 0.0
            return float(np.polyfit(np.arange(len(x)), x.values, 1)[0])

        var_slope = _trend(var.tail(window))
        ac_slope = _trend(ac1.tail(window))
        csd_score = float(np.clip((max(var_slope, 0) * 1000 + max(ac_slope, 0) * 5), 0, 1))
        return {
            "variance_trend": "INCREASING" if var_slope > 0 else "DECREASING",
            "autocorrelation_trend": "INCREASING" if ac_slope > 0 else "DECREASING",
            "critical_slowing_down_score": csd_score,
            "crash_probability": csd_score,
        }

    def calculate_early_warning_signals(self, market_data: dict) -> dict:
        signals = []
        score = 0.0
        if market_data.get("vix_backwardation", 0) > 0.20:
            signals.append("VIX en backwardation pronunciada")
            score += 0.3
        if market_data.get("credit_spread_change_1m", 0) > 0.50:
            signals.append("Credit spreads +50% en 1 mes")
            score += 0.25
        if market_data.get("yield_curve_slope", 1) < 0:
            signals.append("Curva de tipos invertida")
            score += 0.25
        if market_data.get("put_call_ratio", 0) > 1.2:
            signals.append("Put/Call ratio > 1.2")
            score += 0.2
        score = float(np.clip(score, 0, 1))
        level = "EXTREME" if score > 0.75 else "HIGH" if score > 0.5 else "MEDIUM" if score > 0.25 else "LOW"
        return {
            "warning_signals": signals,
            "overall_risk_level": level,
            "crash_probability": score,
            "expected_crash_magnitude": float(-0.1 - 0.2 * score),
            "recommended_hedging": ["long puts", "long VIX", "reduce beta"] if score > 0.5 else [],
        }

    def simulate_crash_scenarios(self, portfolio: dict, shock_magnitude: float = -0.20,
                                 betas: dict | None = None) -> dict:
        betas = betas or {a: 1.0 for a in portfolio}
        impacts = {}
        total = 0.0
        for asset, weight in portfolio.items():
            beta = betas.get(asset, 1.0)
            impact = weight * beta * shock_magnitude
            impacts[asset] = float(impact)
            total += impact
        vulnerable = sorted(impacts, key=impacts.get)[:3]
        return {
            "portfolio_impact": float(total),
            "asset_impacts": impacts,
            "max_drawdown": float(min(total, shock_magnitude)),
            "recovery_time": int(abs(total) * 100),
            "most_vulnerable_assets": vulnerable,
            "hedging_recommendations": ["comprar puts OTM", "rotar a defensivos", "subir cash"],
        }
