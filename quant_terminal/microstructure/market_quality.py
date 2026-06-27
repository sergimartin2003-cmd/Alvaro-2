"""Monitor de calidad de mercado (PROMPT 3.15).

Effective spread, realized spread (= effective - price impact) y monitoreo
agregado de métricas de calidad. numpy/pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class MarketQualityMonitor:
    def calculate_effective_spread(self, trades: pd.DataFrame, quotes: pd.DataFrame) -> pd.Series:
        """Effective spread = 2 * |precio - mid|."""
        mid = ((quotes["bid"] + quotes["ask"]) / 2).reindex(trades.index, method="ffill")
        eff = 2 * (trades["price"] - mid).abs()
        return eff.rename("effective_spread")

    def calculate_realized_spread(self, trades: pd.DataFrame, quotes: pd.DataFrame,
                                  horizon: int = 5) -> pd.Series:
        """Realized spread = effective - price impact (mid futuro vs mid actual)."""
        mid = ((quotes["bid"] + quotes["ask"]) / 2).reindex(trades.index, method="ffill")
        future_mid = mid.shift(-horizon)
        sign = np.sign(trades["price"] - mid).replace(0, 1)
        eff = 2 * (trades["price"] - mid).abs()
        impact = 2 * sign * (future_mid - mid)
        realized = eff - impact.fillna(0)
        return realized.rename("realized_spread")

    def monitor_market_quality_metrics(self, trades: pd.DataFrame, quotes: pd.DataFrame) -> dict:
        eff = self.calculate_effective_spread(trades, quotes)
        realized = self.calculate_realized_spread(trades, quotes)
        mid = ((quotes["bid"] + quotes["ask"]) / 2).reindex(trades.index, method="ffill")
        eff_bps = (eff / mid * 10000).mean()
        metrics = {
            "avg_effective_spread_bps": float(eff_bps),
            "avg_realized_spread": float(realized.mean()),
            "price_impact": float((eff - realized).mean()),
            "quoted_spread_bps": float(((quotes["ask"] - quotes["bid"]) / ((quotes["bid"] + quotes["ask"]) / 2) * 10000).mean()),
        }
        score = float(np.clip(100 - metrics["avg_effective_spread_bps"], 0, 100))
        anomalies = []
        if metrics["avg_effective_spread_bps"] > 50:
            anomalies.append("spread efectivo elevado")
        return {
            "quality_metrics": metrics,
            "quality_score": score,
            "quality_trend": "STABLE",
            "anomalies": anomalies,
        }
