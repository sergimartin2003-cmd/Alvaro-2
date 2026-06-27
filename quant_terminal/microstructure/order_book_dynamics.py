"""Dinámica del order book (PROMPT 3.4).

Tasas de llegada/cancelación/ejecución, patrones de cancelación (spoofing) y
predicción de la evolución del libro. numpy/pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class OrderBookDynamics:
    def model_queue_dynamics(self, order_book_updates: pd.DataFrame) -> dict:
        """Estima tasas por tipo de acción a partir del log de updates."""
        actions = order_book_updates["action"].value_counts(normalize=True).to_dict()
        n = len(order_book_updates)
        arrivals = actions.get("add", 0.0)
        cancels = actions.get("cancel", 0.0)
        execs = actions.get("execute", 0.0)
        exec_prob = execs / (execs + cancels) if (execs + cancels) else 0.0
        return {
            "arrival_rate": float(arrivals),
            "cancelation_rate": float(cancels),
            "execution_rate": float(execs),
            "execution_probability": float(exec_prob),
            "n_updates": n,
        }

    def detect_cancelation_patterns(self, order_book_updates: pd.DataFrame,
                                    fast_ms: int = 500) -> dict:
        cancels = order_book_updates[order_book_updates["action"] == "cancel"]
        total = len(order_book_updates)
        cancel_rate = len(cancels) / total if total else 0.0
        fast_cancels = cancels[cancels.get("lifetime_ms", pd.Series(dtype=float)) < fast_ms]
        suspicious = fast_cancels[fast_cancels.get("volume", 0) > order_book_updates["volume"].mean() * 3]
        spoof_prob = float(len(suspicious) / max(len(cancels), 1))
        return {
            "cancelation_rate": float(cancel_rate),
            "fast_cancelations": int(len(fast_cancels)),
            "suspicious_patterns": suspicious.index.tolist(),
            "spoofing_probability": spoof_prob,
            "manipulation_indicators": ["fast_large_cancels"] if spoof_prob > 0.3 else [],
        }

    def predict_order_book_evolution(self, current_book: dict, imbalance: float = 0.0,
                                     horizon_ms: int = 1000) -> dict:
        """Predicción simple: el mid se desplaza en la dirección del imbalance."""
        bids, asks = current_book["bids"], current_book["asks"]
        mid = (bids[0][0] + asks[0][0]) / 2
        spread = asks[0][0] - bids[0][0]
        expected_change = imbalance * spread * 0.5
        return {
            "predicted_mid": float(mid + expected_change),
            "expected_mid_price_change": float(expected_change),
            "confidence_interval": (float(mid + expected_change - spread),
                                    float(mid + expected_change + spread)),
            "liquidity_forecast": "STABLE" if abs(imbalance) < 0.3 else "SHIFTING",
        }
