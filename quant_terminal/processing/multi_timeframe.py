"""Análisis de confluencia multi-timeframe."""

from __future__ import annotations

import pandas as pd


class MultiTimeframeAnalyzer:
    """Pondera señales por timeframe (D1 > H4 > H1 > M15) y valida timing."""

    def __init__(self) -> None:
        self.timeframes = ["D1", "H4", "H1", "M15"]
        self.weights = {"D1": 0.40, "H4": 0.30, "H1": 0.20, "M15": 0.10}

    def analyze_confluence(self, signals_by_timeframe: dict) -> dict:
        score = 0.0
        counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
        for tf, signal in signals_by_timeframe.items():
            w = self.weights.get(tf, 0.0)
            if signal == "BUY":
                score += w
                counts["BUY"] += 1
            elif signal == "SELL":
                score -= w
                counts["SELL"] += 1
            else:
                counts["HOLD"] += 1

        if score > 0.5 and counts["BUY"] >= 3:
            final = "STRONG_BUY"
        elif score > 0.3:
            final = "BUY"
        elif score < -0.5 and counts["SELL"] >= 3:
            final = "STRONG_SELL"
        elif score < -0.3:
            final = "SELL"
        else:
            final = "HOLD"

        n = max(len(signals_by_timeframe), 1)
        return {
            "final_signal": final,
            "confluence_score": score,
            "signal_counts": counts,
            "agreement_level": max(counts.values()) / n,
        }

    def is_pullback_detected(self, data: pd.DataFrame) -> bool:
        rsi = data["rsi"]
        if rsi.iloc[-1] < 40 and rsi.iloc[-2] < rsi.iloc[-3]:
            return True
        if "ema_20" in data and data["close"].iloc[-1] < data["ema_20"].iloc[-1] * 1.01:
            return True
        return False

    def is_bounce_detected(self, data: pd.DataFrame) -> bool:
        rsi = data["rsi"]
        if rsi.iloc[-1] > 60 and rsi.iloc[-2] > rsi.iloc[-3]:
            return True
        if "ema_20" in data and data["close"].iloc[-1] > data["ema_20"].iloc[-1] * 0.99:
            return True
        return False

    def validate_entry_timing(self, higher_tf_signal: str, lower_tf_data: pd.DataFrame):
        if higher_tf_signal in ("BUY", "STRONG_BUY"):
            if self.is_pullback_detected(lower_tf_data):
                return True, "Pullback detectado - buena entrada"
            return False, "Sin pullback - esperar"
        if higher_tf_signal in ("SELL", "STRONG_SELL"):
            if self.is_bounce_detected(lower_tf_data):
                return True, "Bounce detectado - buena entrada"
            return False, "Sin bounce - esperar"
        return False, "Sin señal en timeframe superior"
