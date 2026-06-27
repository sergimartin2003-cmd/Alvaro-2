"""Detección de shocks de liquidez (PROMPT 1.15).

Detecta shocks (spread, volumen, impacto, TED spread) y puntúa la liquidez de
un activo con grade y slippage estimado. numpy/pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class LiquidityShockDetector:
    def detect_liquidity_shocks(self, market_data: dict) -> list[dict]:
        out = []
        spread = market_data.get("spread")
        avg_spread = market_data.get("avg_spread")
        if spread and avg_spread and spread > 5 * avg_spread:
            out.append({"type": "SPREAD_BLOWOUT", "severity": "HIGH",
                        "affected_assets": market_data.get("symbols", []),
                        "duration_estimate": "horas",
                        "trading_implication": "evitar órdenes a mercado"})
        volume = market_data.get("volume")
        avg_volume = market_data.get("avg_volume")
        if volume is not None and avg_volume and volume < 0.2 * avg_volume:
            out.append({"type": "VOLUME_COLLAPSE", "severity": "HIGH",
                        "affected_assets": market_data.get("symbols", []),
                        "duration_estimate": "sesión",
                        "trading_implication": "reducir tamaño de orden"})
        impact = market_data.get("market_impact")
        normal_impact = market_data.get("normal_impact")
        if impact and normal_impact and impact > 2 * normal_impact:
            out.append({"type": "IMPACT_SPIKE", "severity": "MEDIUM",
                        "affected_assets": market_data.get("symbols", []),
                        "duration_estimate": "minutos",
                        "trading_implication": "usar algos pasivos"})
        ted = market_data.get("ted_spread")
        if ted is not None and ted > 1.0:
            out.append({"type": "FUNDING_STRESS", "severity": "EXTREME",
                        "affected_assets": ["MARKET"],
                        "duration_estimate": "días",
                        "trading_implication": "riesgo sistémico de liquidez"})
        return out

    def calculate_liquidity_score(self, data: dict) -> dict:
        spread_bps = data.get("spread_bps", 10.0)
        daily_volume = data.get("daily_volume", 1e6)
        volatility = data.get("volatility", 0.2)

        score = float(np.clip(
            100 - spread_bps * 2 + min(np.log10(max(daily_volume, 1)) * 5, 30) - volatility * 50,
            0, 100,
        ))
        grade = "A" if score >= 80 else "B" if score >= 60 else "C" if score >= 40 else "D" if score >= 20 else "F"
        slippage = float(spread_bps / 10000 + volatility / np.sqrt(max(daily_volume, 1)))
        return {
            "liquidity_score": score,
            "liquidity_grade": grade,
            "estimated_slippage": slippage,
            "recommended_order_size": float(daily_volume * 0.01),
            "best_execution_time": "apertura/cierre (mayor volumen)",
        }
