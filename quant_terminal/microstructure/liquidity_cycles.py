"""Ciclos de liquidez (PROMPT 3.11).

Patrón intradía de liquidez, detección de shocks y predicción del régimen de
liquidez. numpy/pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class LiquidityCycleAnalyzer:
    def analyze_intraday_liquidity_pattern(self, liquidity_data: pd.DataFrame,
                                           bucket_col: str = "minute_of_day",
                                           liq_col: str = "liquidity") -> dict:
        """Promedia la liquidez por bucket intradía para hallar el patrón en U."""
        pattern = liquidity_data.groupby(bucket_col)[liq_col].mean()
        best = pattern.nlargest(3).index.tolist()
        worst = pattern.nsmallest(3).index.tolist()
        return {
            "intraday_pattern": pattern,
            "best_execution_times": best,
            "worst_execution_times": worst,
            "liquidity_forecast": pattern,
        }

    def detect_liquidity_shocks(self, liquidity_data: pd.Series, threshold: float = 2.0) -> list[dict]:
        s = pd.Series(liquidity_data).dropna()
        z = (s - s.rolling(20, min_periods=1).mean()) / (s.rolling(20, min_periods=1).std() + 1e-9)
        out = []
        for ts, zval in z.items():
            if zval < -threshold:  # caída brusca de liquidez
                out.append({
                    "timestamp": ts,
                    "magnitude": float(zval),
                    "severity": "EXTREME" if zval < -3 else "HIGH",
                    "market_impact": "slippage elevado esperado",
                })
        return out

    def predict_liquidity_regime(self, market_data: dict) -> dict:
        liq = market_data.get("current_liquidity", 50)
        avg = market_data.get("avg_liquidity", 50)
        ratio = liq / avg if avg else 1.0
        if ratio > 1.3:
            regime = "HIGH_LIQUIDITY"
        elif ratio > 0.7:
            regime = "NORMAL_LIQUIDITY"
        elif ratio > 0.4:
            regime = "LOW_LIQUIDITY"
        else:
            regime = "LIQUIDITY_CRISIS"
        return {
            "current_regime": regime,
            "liquidity_ratio": float(ratio),
            "expected_duration": int(market_data.get("persistence", 5)),
            "trading_implications": ["reducir tamaño", "usar algos pasivos"] if "LOW" in regime or "CRISIS" in regime else ["normal"],
        }
