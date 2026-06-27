"""Clasificación de trades (PROMPT 3.5).

Lee-Ready, tick rule y Order Flow Imbalance (OFI). numpy/pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class TradeClassification:
    def classify_trades_lee_ready(self, trades: pd.DataFrame, quotes: pd.DataFrame) -> pd.Series:
        """Compara el precio del trade con el mid; empata con tick rule."""
        mid = ((quotes["bid"] + quotes["ask"]) / 2).reindex(trades.index, method="ffill")
        price = trades["price"]
        labels = pd.Series(index=trades.index, dtype=object)
        labels[price > mid] = "BUY"
        labels[price < mid] = "SELL"
        # Empates: tick rule.
        tie = price == mid
        if tie.any():
            tick = self.classify_trades_tick_rule(trades)
            labels[tie] = tick[tie]
        return labels.fillna("BUY").rename("side")

    def classify_trades_tick_rule(self, trades: pd.DataFrame) -> pd.Series:
        diff = trades["price"].diff()
        labels = pd.Series(index=trades.index, dtype=object)
        labels[diff > 0] = "BUY"
        labels[diff < 0] = "SELL"
        # Cero-tick: hereda la última clasificación.
        labels = labels.ffill().fillna("BUY")
        return labels.rename("side")

    def calculate_order_flow_imbalance(self, classified_trades: pd.DataFrame,
                                       window: int | None = None) -> pd.Series:
        sign = np.where(classified_trades["side"] == "BUY", 1, -1)
        signed_vol = pd.Series(sign * classified_trades["volume"].values, index=classified_trades.index)
        if window:
            return signed_vol.rolling(window, min_periods=1).sum().rename("ofi")
        return signed_vol.cumsum().rename("ofi")
