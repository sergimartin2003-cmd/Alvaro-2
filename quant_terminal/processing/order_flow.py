"""Microestructura de mercado: Volume Profile, delta y absorción."""

from __future__ import annotations

import numpy as np
import pandas as pd


class OrderFlowAnalyzer:
    """Volume Profile (POC/VAH/VAL), delta de agresión y detección de absorción."""

    def calculate_volume_profile(self, tick_data: pd.DataFrame, price_bins: int = 100) -> dict:
        prices = tick_data["price"].values
        volumes = tick_data["volume"].values
        edges = np.linspace(prices.min(), prices.max(), price_bins)
        profile: dict[float, float] = {}
        for i in range(len(edges) - 1):
            mask = (prices >= edges[i]) & (prices < edges[i + 1])
            center = (edges[i] + edges[i + 1]) / 2
            profile[center] = float(volumes[mask].sum())

        poc = max(profile, key=profile.get)
        total = sum(profile.values())
        target = total * 0.70
        cum, va_prices = 0.0, []
        for price, vol in sorted(profile.items(), key=lambda x: x[1], reverse=True):
            cum += vol
            va_prices.append(price)
            if cum >= target:
                break
        return {
            "poc": poc,
            "vah": max(va_prices),
            "val": min(va_prices),
            "volume_profile": profile,
        }

    def calculate_delta(self, tick_data: pd.DataFrame) -> dict:
        buy = tick_data[tick_data["side"] == "buy"]["volume"].sum()
        sell = tick_data[tick_data["side"] == "sell"]["volume"].sum()
        signed = tick_data["volume"] * np.where(tick_data["side"] == "buy", 1, -1)
        return {
            "delta": float(buy - sell),
            "buy_volume": float(buy),
            "sell_volume": float(sell),
            "cum_delta": signed.cumsum(),
        }

    def detect_absorption(self, tick_data: pd.DataFrame, volume_profile: dict) -> dict:
        poc = volume_profile["poc"]
        near = tick_data[(tick_data["price"] - poc).abs() < poc * 0.001]
        if len(near) > 0:
            total = float(near["volume"].sum())
            price_range = (near["price"].max() - near["price"].min()) / poc
            if total > 10000 and price_range < 0.001:
                return {
                    "absorption_detected": True,
                    "volume": total,
                    "price_range": price_range,
                    "signal": "STRONG_SUPPORT_OR_RESISTANCE",
                }
        return {"absorption_detected": False}
