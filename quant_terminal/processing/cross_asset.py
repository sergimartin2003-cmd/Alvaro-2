"""Análisis cross-asset: correlaciones rolling, cambios de régimen, risk appetite."""

from __future__ import annotations

import pandas as pd


class CrossAssetAnalyzer:
    """Correlaciones entre clases de activos y apetito de riesgo."""

    def __init__(self) -> None:
        self.assets = {
            "equities": ["SPY", "QQQ", "IWM", "EFA", "EEM"],
            "bonds": ["TLT", "IEF", "SHY", "HYG", "LQD"],
            "commodities": ["GLD", "SLV", "USO", "DBA", "DBB"],
            "currencies": ["UUP", "FXE", "FXJ", "FXB"],
            "volatility": ["VXX", "UVXY", "SVXY"],
        }

    def calculate_rolling_correlations(
        self, price_data: pd.DataFrame, window: int = 60
    ) -> pd.DataFrame:
        returns = price_data.pct_change().dropna()
        cols = list(returns.columns)
        out = {}
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                out[f"{a}_{b}"] = returns[a].rolling(window).corr(returns[b])
        return pd.DataFrame(out)

    def detect_regime_change(self, correlations: pd.DataFrame, threshold: float = 0.3) -> dict:
        changes = {}
        for pair in correlations.columns:
            series = correlations[pair]
            delta = series.diff(20)
            if abs(delta.iloc[-1]) > threshold:
                changes[pair] = {
                    "current_corr": float(series.iloc[-1]),
                    "corr_change": float(delta.iloc[-1]),
                    "direction": "INCREASING" if delta.iloc[-1] > 0 else "DECREASING",
                }
        return changes

    def calculate_risk_appetite_index(self, price_data: pd.DataFrame) -> dict:
        returns = price_data.pct_change().dropna()
        risk_on = [c for c in ["SPY", "QQQ", "IWM", "GLD", "USO"] if c in returns]
        risk_off = [c for c in ["TLT", "UUP", "SHY"] if c in returns]
        if not risk_on or not risk_off:
            raise ValueError("Faltan activos risk-on/risk-off en los datos")
        on = returns[risk_on].mean(axis=1).rolling(20).mean()
        off = returns[risk_off].mean(axis=1).rolling(20).mean()
        appetite = (on - off).dropna()
        rng = appetite.max() - appetite.min()
        score = ((appetite - appetite.min()) / rng * 100) if rng else appetite * 0
        return {
            "score": float(score.iloc[-1]),
            "trend": "INCREASING" if score.diff(5).iloc[-1] > 0 else "DECREASING",
            "regime": "RISK-ON" if score.iloc[-1] > 60 else "RISK-OFF",
        }
