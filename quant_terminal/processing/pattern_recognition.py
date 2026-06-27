"""Reconocimiento de patrones de velas, patrones gráficos y divergencias."""

from __future__ import annotations

import numpy as np
import pandas as pd


class PatternRecognizer:
    """Detección de candlesticks, soportes/resistencias y divergencias RSI."""

    def detect_candlestick_patterns(self, df: pd.DataFrame) -> list[dict]:
        patterns = []
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        for i in range(1, len(df)):
            body = abs(c.iloc[i] - o.iloc[i])
            rng = h.iloc[i] - l.iloc[i]
            if rng == 0:
                continue
            lower_wick = min(o.iloc[i], c.iloc[i]) - l.iloc[i]
            upper_wick = h.iloc[i] - max(o.iloc[i], c.iloc[i])
            ts = str(df.index[i])

            # Doji
            if body <= rng * 0.1:
                patterns.append({"pattern": "DOJI", "timestamp": ts, "strength": 0.5})
            # Hammer (martillo alcista)
            if lower_wick > body * 2 and upper_wick < body and c.iloc[i] > o.iloc[i]:
                patterns.append({"pattern": "HAMMER", "timestamp": ts, "strength": 0.7})
            # Shooting star
            if upper_wick > body * 2 and lower_wick < body and c.iloc[i] < o.iloc[i]:
                patterns.append({"pattern": "SHOOTING_STAR", "timestamp": ts, "strength": 0.7})
            # Engulfing
            prev_body = abs(c.iloc[i - 1] - o.iloc[i - 1])
            if c.iloc[i] > o.iloc[i] and c.iloc[i - 1] < o.iloc[i - 1] and body > prev_body:
                patterns.append({"pattern": "BULLISH_ENGULFING", "timestamp": ts, "strength": 0.8})
            if c.iloc[i] < o.iloc[i] and c.iloc[i - 1] > o.iloc[i - 1] and body > prev_body:
                patterns.append({"pattern": "BEARISH_ENGULFING", "timestamp": ts, "strength": 0.8})
        return patterns

    def detect_divergences(self, df: pd.DataFrame, lookback: int = 30) -> list[dict]:
        if "rsi" not in df.columns:
            return []
        window = df.tail(lookback)
        price, rsi = window["close"].values, window["rsi"].values
        out = []
        # Compara mínimos/máximos de la primera y segunda mitad de la ventana.
        half = len(window) // 2
        if half < 2:
            return out
        p1_min, p2_min = price[:half].min(), price[half:].min()
        r1_min, r2_min = rsi[:half].min(), rsi[half:].min()
        if p2_min < p1_min and r2_min > r1_min:
            out.append({"type": "BULLISH_DIVERGENCE", "strength": 0.75})
        p1_max, p2_max = price[:half].max(), price[half:].max()
        r1_max, r2_max = rsi[:half].max(), rsi[half:].max()
        if p2_max > p1_max and r2_max < r1_max:
            out.append({"type": "BEARISH_DIVERGENCE", "strength": 0.75})
        return out
