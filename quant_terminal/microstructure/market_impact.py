"""Modelo de impacto de mercado (PROMPT 3.2).

Estimación de la square-root law, ejecución óptima Almgren-Chriss y simulación
de impacto. numpy/pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class MarketImpactModel:
    def estimate_square_root_law_parameters(self, trades: pd.DataFrame,
                                            daily_volume: float = 1e7) -> dict:
        """Impact ≈ Y * sigma * sqrt(Q / V). Ajusta Y por regresión."""
        df = trades.dropna(subset=["size", "impact"]) if "impact" in trades else trades
        sigma = float(np.log(trades["price"] / trades["price"].shift(1)).std()) if "price" in trades else 0.02
        if "impact" in df and "size" in df and len(df) > 2:
            x = np.sqrt(df["size"].values / daily_volume)
            y = df["impact"].values
            Y = float(np.polyfit(x * sigma, y, 1)[0]) if np.std(x) > 0 else 0.5
            pred = Y * sigma * x
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - np.sum((y - pred) ** 2) / ss_tot if ss_tot else 0.0
        else:
            Y, r2 = 0.5, 0.0
        return {"Y": Y, "sigma": sigma, "r_squared": float(r2)}

    def calculate_optimal_execution(self, order_size: float, horizon: int,
                                    risk_aversion: float = 1.0, sigma: float = 0.02,
                                    eta: float = 1e-6) -> dict:
        """Almgren-Chriss: trayectoria óptima con decaimiento hiperbólico."""
        kappa = np.sqrt(risk_aversion * sigma**2 / max(eta, 1e-12))
        t = np.arange(horizon + 1)
        if kappa * horizon > 1e-6:
            remaining = order_size * np.sinh(kappa * (horizon - t)) / np.sinh(kappa * horizon)
        else:
            remaining = order_size * (1 - t / horizon)
        schedule = -np.diff(remaining)
        expected_cost = float(0.5 * eta * np.sum(schedule**2))
        return {
            "execution_schedule": schedule.tolist(),
            "remaining_trajectory": remaining.tolist(),
            "expected_cost": expected_cost,
            "cost_std": float(sigma * np.sqrt(np.sum(remaining[:-1] ** 2))),
            "optimal_participation_rate": float(schedule.mean() / order_size) if order_size else 0.0,
        }

    def simulate_market_impact(self, order_size: float, daily_volume: float = 1e7,
                               sigma: float = 0.02, Y: float = 0.5) -> dict:
        ratio = order_size / max(daily_volume, 1)
        total = Y * sigma * np.sqrt(ratio)
        permanent = total * 0.4
        temporary = total * 0.6
        steps = 10
        traj = np.cumsum(np.full(steps, permanent / steps)) + temporary * np.exp(-np.arange(steps) / 3)
        return {
            "permanent_impact": float(permanent),
            "temporary_impact": float(temporary),
            "total_impact_bps": float(total * 10000),
            "price_trajectory": traj.tolist(),
            "execution_cost": float(total * order_size),
        }
