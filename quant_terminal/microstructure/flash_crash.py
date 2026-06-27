"""Detección de flash crashes (PROMPT 3.10).

Detección (caída fuerte en ventana corta), análisis de dinámica y predicción de
riesgo. numpy/pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class FlashCrashDetector:
    def detect_flash_crash(self, prices: pd.Series, window: int = 5,
                           threshold_pct: float = -5.0) -> dict:
        p = pd.Series(prices).reset_index(drop=True)
        roll_ret = (p / p.shift(window) - 1) * 100
        min_ret = roll_ret.min()
        is_crash = bool(min_ret <= threshold_pct)
        crash_idx = int(roll_ret.idxmin()) if is_crash else None

        recovery = 0
        if is_crash and crash_idx is not None:
            trough = p.iloc[crash_idx]
            pre = p.iloc[max(crash_idx - window, 0)]
            for j in range(crash_idx, len(p)):
                if p.iloc[j] >= pre * 0.99:
                    recovery = j - crash_idx
                    break
        severity = "EXTREME" if min_ret <= -10 else "HIGH" if min_ret <= -7 else "MEDIUM"
        return {
            "is_flash_crash": is_crash,
            "crash_magnitude": float(min_ret) if np.isfinite(min_ret) else 0.0,
            "crash_index": crash_idx,
            "recovery_time": recovery,
            "severity": severity if is_crash else "NONE",
        }

    def analyze_flash_crash_dynamics(self, prices: pd.Series, volume: pd.Series | None = None) -> dict:
        crash = self.detect_flash_crash(prices)
        amplifiers = []
        if volume is not None and len(volume) > 5:
            v = pd.Series(volume)
            if v.iloc[-1] > 3 * v.mean():
                amplifiers.append("volume_spike")
        p = pd.Series(prices)
        recovery_pattern = "V_SHAPED" if crash["recovery_time"] and crash["recovery_time"] < 10 else "PROLONGED"
        return {
            "crash": crash,
            "amplification_factors": amplifiers,
            "recovery_pattern": recovery_pattern if crash["is_flash_crash"] else "N/A",
            "trading_opportunities": ["mean reversion tras el trough"] if crash["is_flash_crash"] else [],
        }

    def predict_flash_crash_risk(self, market_data: dict) -> dict:
        score = 0.0
        factors = []
        if market_data.get("liquidity_score", 100) < 30:
            score += 0.4
            factors.append("liquidez baja")
        if market_data.get("volatility", 0) > 0.4:
            score += 0.3
            factors.append("volatilidad extrema")
        if market_data.get("order_imbalance", 0) > 0.7:
            score += 0.3
            factors.append("imbalance extremo")
        score = float(np.clip(score, 0, 1))
        return {
            "crash_risk_score": score,
            "risk_factors": factors,
            "probability_next_hour": score * 0.5,
            "recommended_actions": ["reducir tamaño", "stops dinámicos"] if score > 0.5 else [],
        }
