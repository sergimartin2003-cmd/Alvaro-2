"""Predicción de order flow (PROMPT 3.12).

Predicción de OFI (AR), forecast de volumen (estacional) y tasa de llegada de
órdenes (proceso de Hawkes simplificado). numpy/pandas con statsmodels opcional.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class OrderFlowPredictor:
    def predict_order_flow_imbalance(self, historical_ofi: pd.Series, horizon: int = 10) -> pd.Series:
        """Predicción AR(1) del OFI (con fallback a persistencia)."""
        s = pd.Series(historical_ofi).dropna()
        if len(s) < 3:
            return pd.Series([float(s.iloc[-1]) if len(s) else 0.0] * horizon)
        x = s.values[:-1]
        y = s.values[1:]
        phi = float(np.polyfit(x, y, 1)[0]) if np.std(x) > 0 else 0.0
        phi = np.clip(phi, -0.99, 0.99)
        mean = s.mean()
        last = s.iloc[-1]
        preds = []
        for _ in range(horizon):
            last = mean + phi * (last - mean)
            preds.append(last)
        return pd.Series(preds, name="ofi_forecast")

    def forecast_volume(self, historical_volume: pd.Series, horizon: int = 5,
                        season: int = 5) -> pd.Series:
        """Forecast de volumen por descomposición estacional simple."""
        s = pd.Series(historical_volume).dropna()
        if len(s) < season * 2:
            return pd.Series([float(s.mean())] * horizon)
        seasonal = s.groupby(np.arange(len(s)) % season).mean()
        trend = s.rolling(season, min_periods=1).mean().iloc[-1]
        base = trend
        preds = [float(base * (seasonal.iloc[i % season] / seasonal.mean())) for i in range(horizon)]
        return pd.Series(preds, name="volume_forecast")

    def predict_order_arrival_rate(self, event_times: np.ndarray, horizon_ms: int = 1000,
                                   decay: float = 0.01, excitation: float = 0.5) -> dict:
        """Hawkes simplificado: intensidad = mu + sum(excitation * exp(-decay*dt))."""
        t = np.asarray(event_times, dtype=float)
        if len(t) < 2:
            return {"predicted_arrival_rate": 0.0, "confidence_intervals": (0.0, 0.0)}
        now = t[-1]
        mu = len(t) / (now - t[0] + 1e-9)
        intensity = mu + np.sum(excitation * np.exp(-decay * (now - t)))
        expected = intensity * horizon_ms / 1000
        return {
            "predicted_arrival_rate": float(intensity),
            "expected_events_in_horizon": float(expected),
            "confidence_intervals": (float(expected * 0.7), float(expected * 1.3)),
        }
