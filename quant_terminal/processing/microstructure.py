"""Microestructura de mercado (PROMPT 1.3).

VPIN, impacto de precio (square-root law), flujo tóxico, spread óptimo
(Avellaneda-Stoikov) y detección de trading informado. numpy/pandas/scipy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


class MarketMicrostructureAnalyzer:
    def calculate_VPIN(self, trades: pd.DataFrame, bucket_size: int = 50) -> pd.Series:
        """Volume-Synchronized Probability of Informed Trading."""
        if "side" in trades.columns:
            signed = np.where(trades["side"] == "buy", 1, -1)
        else:  # tick rule
            signed = np.sign(trades["price"].diff().fillna(0)).replace(0, 1).values
        vol = trades["volume"].values
        buy_v = np.where(signed > 0, vol, 0.0)
        sell_v = np.where(signed < 0, vol, 0.0)

        cum = np.cumsum(vol)
        n_buckets = int(cum[-1] // bucket_size) if len(cum) else 0
        vpins = []
        start = 0
        for b in range(1, n_buckets + 1):
            end = np.searchsorted(cum, b * bucket_size)
            bv, sv = buy_v[start:end].sum(), sell_v[start:end].sum()
            vpins.append(abs(bv - sv) / bucket_size)
            start = end
        s = pd.Series(vpins, name="vpin")
        return s.rolling(5, min_periods=1).mean()

    def estimate_price_impact(self, order_size, symbol=None, side="buy",
                              sigma: float = 0.02, daily_volume: float = 1e7,
                              Y: float = 0.5) -> dict:
        """Impacto = Y * sigma * sqrt(Q / V)."""
        ratio = order_size / max(daily_volume, 1)
        total = Y * sigma * np.sqrt(ratio)
        permanent = total * 0.4
        temporary = total * 0.6
        if ratio > 0.05:
            algo = "VWAP"
        elif ratio > 0.02:
            algo = "TWAP"
        else:
            algo = "ICEBERG" if ratio > 0.005 else "MARKET"
        return {
            "permanent_impact": float(permanent),
            "temporary_impact": float(temporary),
            "total_impact_bps": float(total * 10000),
            "recommended_algo": algo,
        }

    def detect_toxic_flow(self, trades: pd.DataFrame, order_book=None) -> dict:
        vpin = self.calculate_VPIN(trades)
        score = float(vpin.iloc[-1]) if len(vpin) else 0.0
        toxic_pct = float((vpin > 0.7).mean()) if len(vpin) else 0.0
        if score > 0.7:
            rec = "STOP_MARKET_MAKING"
        elif score > 0.5:
            rec = "WIDEN_SPREAD"
        else:
            rec = "REDUCE_SIZE" if score > 0.4 else "NORMAL"
        return {
            "toxicity_score": score,
            "toxic_trades_pct": toxic_pct,
            "recommendation": rec,
        }

    def calculate_optimal_spread(self, symbol=None, risk_aversion: float = 1.0,
                                 sigma: float = 0.02, T: float = 1.0, t: float = 0.0,
                                 inventory: float = 0.0, max_inventory: float = 100.0,
                                 mid_price: float = 100.0) -> dict:
        """Avellaneda-Stoikov."""
        tau = max(T - t, 1e-6)
        gamma = risk_aversion
        q = inventory / max(max_inventory, 1)
        spread = gamma * sigma**2 * tau + 2 * gamma * sigma * np.sqrt(tau) * norm.ppf(
            min(max(1 - abs(q), 1e-3), 1 - 1e-3)
        )
        spread = abs(spread)
        reservation = mid_price - q * gamma * sigma**2 * tau
        return {
            "optimal_spread_bps": float(spread / mid_price * 10000),
            "optimal_quote": {"bid": float(reservation - spread / 2),
                              "ask": float(reservation + spread / 2)},
            "reservation_price": float(reservation),
            "expected_profit_per_trade": float(spread / 2),
        }

    def detect_informed_trading(self, trades: pd.DataFrame) -> dict:
        vpin = self.calculate_VPIN(trades)
        score = float(vpin.mean()) if len(vpin) else 0.0
        # Trades grandes que preceden movimientos fuertes = sospechosos.
        suspicious = []
        if "volume" in trades and len(trades) > 5:
            big = trades[trades["volume"] > trades["volume"].quantile(0.95)]
            suspicious = big.index.tolist()
        return {
            "informed_trading_score": score,
            "suspicious_trades": suspicious,
            "likely_catalyst": "Posible información no pública" if score > 0.6 else "Ninguno claro",
        }
