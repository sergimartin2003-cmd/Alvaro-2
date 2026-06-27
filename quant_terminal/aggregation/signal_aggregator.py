"""Agregación de señales multi-fuente con ponderación dinámica."""

from __future__ import annotations

import numpy as np


class SignalAggregator:
    """Combina señales de varias fuentes y ajusta pesos por performance."""

    def __init__(self) -> None:
        self.signal_sources = {
            "technical_analysis": {"weight": 0.25, "performance_history": []},
            "sentiment_analysis": {"weight": 0.15, "performance_history": []},
            "macro_analysis": {"weight": 0.20, "performance_history": []},
            "options_flow": {"weight": 0.15, "performance_history": []},
            "ml_models": {"weight": 0.15, "performance_history": []},
            "seasonality": {"weight": 0.10, "performance_history": []},
        }

    def calculate_weighted_score(self, signals_dict: dict) -> float:
        score = 0.0
        for source, data in signals_dict.items():
            if source not in self.signal_sources:
                continue
            w = self.signal_sources[source]["weight"]
            conf = data.get("confidence", 0.5)
            strength = data.get("strength", conf)
            signal = data.get("signal", "HOLD")
            if signal in ("BUY", "STRONG_BUY"):
                s = conf * strength
            elif signal in ("SELL", "STRONG_SELL"):
                s = -conf * strength
            else:
                s = 0.0
            score += w * s
        return float(score)

    def detect_signal_conflicts(self, signals_dict: dict) -> dict:
        buy, sell, hold = [], [], []
        for source, data in signals_dict.items():
            sig = data.get("signal", "HOLD")
            if sig in ("BUY", "STRONG_BUY"):
                buy.append(source)
            elif sig in ("SELL", "STRONG_SELL"):
                sell.append(source)
            else:
                hold.append(source)
        conflict = bool(buy) and bool(sell)
        return {
            "conflict_detected": conflict,
            "buy_sources": buy,
            "sell_sources": sell,
            "hold_sources": hold,
            "conflict_severity": "HIGH" if conflict else "NONE",
            "recommendation": "REDUCE_POSITION_SIZE" if conflict else "NORMAL",
        }

    def aggregate_signals(self, signals_dict: dict) -> dict:
        score = self.calculate_weighted_score(signals_dict)
        conflicts = self.detect_signal_conflicts(signals_dict)
        known = {k: v for k, v in signals_dict.items() if k in self.signal_sources}
        n = max(len(known), 1)
        agreement = max(
            len(conflicts["buy_sources"]), len(conflicts["sell_sources"]), len(conflicts["hold_sources"])
        ) / n

        if score > 0.3:
            signal = "BUY"
        elif score < -0.3:
            signal = "SELL"
        else:
            signal = "HOLD"

        return {
            "aggregate_signal": signal,
            "aggregate_score": score,
            "aggregate_confidence": min(1.0, abs(score) * 2),
            "confluence_analysis": {
                **conflicts,
                "agreement_percentage": agreement,
            },
            "source_contributions": signals_dict,
        }

    def update_source_weights(self, performance_data: dict) -> None:
        sharpe = {}
        for source, data in performance_data.items():
            trades = data.get("wins", 0) + data.get("losses", 0)
            if trades > 0:
                sharpe[source] = data.get("total_pnl", 0.0) / max(0.01, trades)
        total = sum(max(s, 0) for s in sharpe.values())
        if total > 0:
            for source in self.signal_sources:
                if source in sharpe:
                    new_w = max(sharpe[source], 0) / total
                    old_w = self.signal_sources[source]["weight"]
                    self.signal_sources[source]["weight"] = old_w * 0.7 + new_w * 0.3
        else:
            eq = 1.0 / len(self.signal_sources)
            for source in self.signal_sources:
                self.signal_sources[source]["weight"] = eq
