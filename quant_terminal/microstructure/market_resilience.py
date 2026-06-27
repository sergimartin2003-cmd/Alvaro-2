"""Resiliencia de mercado (PROMPT 3.13).

Resiliencia de precio (velocidad de recuperación), replenishment de liquidez y
evaluación de estabilidad. numpy/pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class MarketResilienceAnalyzer:
    def measure_price_resilience(self, prices: pd.Series, shock_threshold: float = 0.005) -> dict:
        """Tras un shock de precio, ¿cuántos pasos tarda en revertir?"""
        p = pd.Series(prices).reset_index(drop=True)
        ret = p.pct_change().fillna(0)
        recoveries = []
        for i in range(1, len(p) - 1):
            if abs(ret.iloc[i]) > shock_threshold:
                pre = p.iloc[i - 1]
                for j in range(i + 1, len(p)):
                    if (ret.iloc[i] > 0 and p.iloc[j] <= pre) or (ret.iloc[i] < 0 and p.iloc[j] >= pre):
                        recoveries.append(j - i)
                        break
        avg_recovery = float(np.mean(recoveries)) if recoveries else 0.0
        score = float(np.clip(100 / (1 + avg_recovery), 0, 100))
        return {
            "resilience_score": score,
            "average_recovery_time": avg_recovery,
            "n_shocks": len(recoveries),
            "market_stability": "STABLE" if score > 60 else "FRAGILE",
        }

    def analyze_liquidity_replenishment(self, depth_series: pd.Series) -> dict:
        """Velocidad con que la profundidad vuelve a su media tras caer."""
        s = pd.Series(depth_series).dropna()
        mean = s.mean()
        below = s < mean * 0.5
        speed = float(1 / (below.sum() + 1)) if len(s) else 0.0
        return {
            "replenishment_speed": speed,
            "replenishment_rate": float((s.diff() > 0).mean()),
            "liquidity_stability": "GOOD" if speed > 0.3 else "POOR",
        }

    def assess_market_stability(self, market_data: dict) -> dict:
        vol = market_data.get("volatility", 0.2)
        resilience = market_data.get("resilience_score", 50)
        spread_stability = market_data.get("spread_stability", 0.5)
        score = float(np.clip(resilience * 0.5 + spread_stability * 50 - vol * 100, 0, 100))
        signals = []
        if vol > 0.4:
            signals.append("volatilidad extrema")
        if resilience < 30:
            signals.append("baja resiliencia")
        return {
            "stability_score": score,
            "stability_regime": "STABLE" if score > 60 else "UNSTABLE" if score < 30 else "TRANSITIONAL",
            "risk_of_instability": float(1 - score / 100),
            "early_warning_signals": signals,
        }
