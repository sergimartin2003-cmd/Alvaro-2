"""Toxicidad del order flow (PROMPT 3.1).

VPIN, adverse selection y clasificación de traders (informados vs no
informados). numpy/pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class OrderFlowToxicityAnalyzer:
    def calculate_VPIN(self, trades: pd.DataFrame, bucket_size: int = 50) -> pd.Series:
        if "side" in trades.columns:
            signed = np.where(trades["side"] == "buy", 1, -1)
        else:
            signed = np.sign(trades["price"].diff().fillna(0)).replace(0, 1).values
        vol = trades["volume"].values
        buy_v = np.where(signed > 0, vol, 0.0)
        sell_v = np.where(signed < 0, vol, 0.0)
        cum = np.cumsum(vol)
        n_buckets = int(cum[-1] // bucket_size) if len(cum) else 0
        out, start = [], 0
        for b in range(1, n_buckets + 1):
            end = int(np.searchsorted(cum, b * bucket_size))
            out.append(abs(buy_v[start:end].sum() - sell_v[start:end].sum()) / bucket_size)
            start = end
        return pd.Series(out, name="vpin").rolling(5, min_periods=1).mean()

    def detect_adverse_selection(self, quotes: pd.DataFrame, trades: pd.DataFrame) -> dict:
        """Adverse selection: el precio se mueve en contra del market maker tras el fill."""
        merged = trades.copy()
        mid = (quotes["bid"] + quotes["ask"]) / 2
        merged = merged.assign(mid=mid.reindex(merged.index, method="ffill"))
        future_mid = merged["mid"].shift(-5)
        # Para fills de compra (taker compra al ask), pérdida del MM si el mid sube.
        sign = np.where(merged.get("side", "buy") == "buy", 1, -1)
        adverse = sign * (future_mid - merged["mid"]).fillna(0)
        loss_per_trade = float(adverse[adverse > 0].mean()) if (adverse > 0).any() else 0.0
        toxic_pct = float((adverse > 0).mean())
        score = float(np.clip(toxic_pct, 0, 1))
        rec = "WIDEN_SPREAD" if score > 0.6 else "REDUCE_SIZE" if score > 0.4 else "NORMAL"
        return {
            "adverse_selection_score": score,
            "estimated_loss_per_trade": loss_per_trade,
            "toxic_flow_percentage": toxic_pct,
            "recommendation": rec,
        }

    def classify_traders(self, trades: pd.DataFrame) -> dict:
        """Clasifica trades como informados (grandes y direccionales) o no."""
        vol = trades["volume"]
        big = vol > vol.quantile(0.9)
        informed = trades[big]
        pin = float(big.mean())  # proxy del Probability of Informed Trading
        return {
            "informed_trades": informed.index.tolist(),
            "uninformed_trades": trades[~big].index.tolist(),
            "pin": pin,
            "informed_flow_percentage": float(informed["volume"].sum() / vol.sum()) if vol.sum() else 0.0,
        }
