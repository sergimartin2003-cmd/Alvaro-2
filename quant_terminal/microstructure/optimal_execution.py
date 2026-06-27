"""Ejecución óptima (PROMPT 3.7).

TWAP, VWAP, Implementation Shortfall (Almgren-Chriss) y POV. numpy/pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class OptimalExecutionEngine:
    def execute_twap(self, total_quantity: float, n_slices: int) -> dict:
        per = total_quantity / n_slices
        schedule = [per] * n_slices
        return {
            "execution_schedule": schedule,
            "child_orders": [{"slice": i, "qty": per} for i in range(n_slices)],
            "expected_slippage": 0.0,
        }

    def execute_vwap(self, total_quantity: float, volume_profile: pd.Series) -> dict:
        v = np.asarray(volume_profile, dtype=float)
        weights = v / v.sum() if v.sum() else np.ones(len(v)) / len(v)
        schedule = (weights * total_quantity).tolist()
        return {
            "execution_schedule": schedule,
            "volume_profile": weights.tolist(),
            "child_orders": [{"slice": i, "qty": q} for i, q in enumerate(schedule)],
            "expected_slippage": 0.0,
        }

    def execute_implementation_shortfall(self, total_quantity: float, horizon: int,
                                         risk_aversion: float = 1.0, sigma: float = 0.02,
                                         eta: float = 1e-6) -> dict:
        kappa = np.sqrt(risk_aversion * sigma**2 / max(eta, 1e-12))
        t = np.arange(horizon + 1)
        if kappa * horizon > 1e-6:
            remaining = total_quantity * np.sinh(kappa * (horizon - t)) / np.sinh(kappa * horizon)
        else:
            remaining = total_quantity * (1 - t / horizon)
        schedule = -np.diff(remaining)
        expected_is = float(0.5 * eta * np.sum(schedule**2))
        return {
            "execution_schedule": schedule.tolist(),
            "expected_is": expected_is,
            "is_variance": float((sigma**2) * np.sum(remaining[:-1] ** 2)),
        }

    def execute_pov(self, total_quantity: float, market_volume_forecast: pd.Series,
                    participation_rate: float = 0.10) -> dict:
        v = np.asarray(market_volume_forecast, dtype=float)
        schedule, remaining = [], total_quantity
        for vol in v:
            q = min(participation_rate * vol, remaining)
            schedule.append(float(q))
            remaining -= q
            if remaining <= 0:
                break
        return {
            "execution_schedule": schedule,
            "participation_rate": participation_rate,
            "completed": remaining <= 1e-9,
            "remaining": float(max(remaining, 0)),
        }
