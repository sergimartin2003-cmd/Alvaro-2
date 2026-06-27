"""Filtro de flujo tóxico (PROMPT 3.14).

Filtra periodos tóxicos por VPIN, detecta trading informado en tiempo real y
ajusta cotizaciones según la toxicidad. numpy/pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .order_flow_toxicity import OrderFlowToxicityAnalyzer


class ToxicityFlowFilter:
    def __init__(self) -> None:
        self._toxicity = OrderFlowToxicityAnalyzer()

    def filter_toxic_flow(self, trades: pd.DataFrame, vpin_threshold: float = 0.7,
                          bucket_size: int = 50) -> dict:
        vpin = self._toxicity.calculate_VPIN(trades, bucket_size=bucket_size)
        toxic = vpin[vpin > vpin_threshold]
        clean = vpin[vpin <= vpin_threshold]
        actions = []
        if len(toxic) / max(len(vpin), 1) > 0.3:
            actions.append("WIDEN_SPREADS")
        if len(vpin) and vpin.iloc[-1] > vpin_threshold:
            actions.append("PAUSE_MARKET_MAKING")
        return {
            "toxic_periods": toxic.index.tolist(),
            "clean_periods": clean.index.tolist(),
            "toxic_fraction": float(len(toxic) / max(len(vpin), 1)),
            "recommended_actions": actions or ["NORMAL"],
        }

    def detect_informed_trading_in_realtime(self, trades: pd.DataFrame, window: int = 50) -> pd.Series:
        """Probabilidad rolling de trading informado (volumen relativo grande)."""
        vol = trades["volume"]
        rel = vol / vol.rolling(window, min_periods=1).mean()
        prob = (rel / (rel + 1)).clip(0, 1)
        return prob.rename("informed_prob")

    def adjust_quotes_for_toxicity(self, current_quotes: dict, toxicity_score: float) -> dict:
        """Ensancha el spread proporcionalmente a la toxicidad."""
        mid = (current_quotes["bid"] + current_quotes["ask"]) / 2
        spread = current_quotes["ask"] - current_quotes["bid"]
        factor = 1 + 2 * np.clip(toxicity_score, 0, 1)
        new_spread = spread * factor
        return {
            "bid": float(mid - new_spread / 2),
            "ask": float(mid + new_spread / 2),
            "spread_widening_factor": float(factor),
        }
