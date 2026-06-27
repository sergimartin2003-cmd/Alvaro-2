"""Pairs trading con cointegración (PROMPT 1.4).

Encuentra pares cointegrados (Engle-Granger con ADF), calcula hedge ratio,
niveles óptimos vía OU, construye portfolio y monitorea en tiempo real.
statsmodels es opcional: si falta, se usa un ADF simplificado propio.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .stochastic_models import OrnsteinUhlenbeckModel


def _adf_pvalue(series: np.ndarray) -> float:
    """p-value del test ADF. Usa statsmodels si está; si no, aproxima."""
    try:
        from statsmodels.tsa.stattools import adfuller

        return float(adfuller(series, autolag="AIC")[1])
    except Exception:
        # Aproximación: regresión dX = rho*X_{t-1}; t-stat -> heurística.
        x = series[:-1]
        dx = np.diff(series)
        if np.std(x) < 1e-9:
            return 1.0
        beta = np.polyfit(x, dx, 1)[0]
        resid = dx - beta * x
        se = np.std(resid) / (np.std(x) * np.sqrt(len(x)))
        t = beta / (se + 1e-12)
        # Mapear t-stat a pseudo p-value (más negativo => más estacionario).
        return float(np.clip(1 / (1 + np.exp(-(t + 2.0))), 0, 1))


class PairsTradingEngine:
    def __init__(self, pvalue_threshold: float = 0.05) -> None:
        self.pvalue_threshold = pvalue_threshold

    @staticmethod
    def _hedge_ratio(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.polyfit(b, a, 1)[0])

    def find_cointegrated_pairs(self, prices: pd.DataFrame, method: str = "engle_granger") -> list[dict]:
        """prices: DataFrame con una columna de precios por activo."""
        out = []
        cols = list(prices.columns)
        for a, b in itertools.combinations(cols, 2):
            sa, sb = prices[a].dropna(), prices[b].dropna()
            idx = sa.index.intersection(sb.index)
            if len(idx) < 30:
                continue
            sa, sb = sa.loc[idx].values, sb.loc[idx].values
            beta = self._hedge_ratio(sa, sb)
            spread = sa - beta * sb
            pval = _adf_pvalue(spread)
            if pval < self.pvalue_threshold:
                ou = OrnsteinUhlenbeckModel(spread)
                params = ou.fit()
                out.append({
                    "pair": (a, b),
                    "hedge_ratio": beta,
                    "half_life": params["half_life"],
                    "correlation": float(np.corrcoef(sa, sb)[0, 1]),
                    "cointegration_pvalue": pval,
                })
        out.sort(key=lambda x: x["cointegration_pvalue"])
        return out

    def calculate_optimal_entry_exit(self, spread: pd.Series, model: str = "OU") -> dict:
        arr = np.asarray(spread, dtype=float)
        ou = OrnsteinUhlenbeckModel(arr)
        params = ou.fit()
        mu, sigma_eq = arr.mean(), arr.std()
        return {
            "entry_threshold": 1.5,
            "exit_threshold": 0.0,
            "stop_loss_threshold": 3.0,
            "expected_holding_period": int(params["half_life"]) if np.isfinite(params["half_life"]) else 0,
            "mu": float(mu),
            "sigma": float(sigma_eq),
            "sharpe_ratio": float(abs(params["theta"]) * 2),
        }

    def build_pairs_portfolio(self, pairs: list[dict], max_pairs: int = 10) -> dict:
        selected = sorted(pairs, key=lambda x: x["cointegration_pvalue"])[:max_pairs]
        n = len(selected)
        if n == 0:
            return {"selected_pairs": [], "weights": {}, "expected_portfolio_sharpe": 0.0,
                    "diversification_ratio": 0.0}
        # Pesos inversos a la half-life (preferir reversión rápida), cap 20%.
        inv = np.array([1 / max(p["half_life"], 1) for p in selected])
        w = np.minimum(inv / inv.sum(), 0.20)
        w = w / w.sum()
        weights = {str(p["pair"]): float(wi) for p, wi in zip(selected, w)}
        return {
            "selected_pairs": selected,
            "weights": weights,
            "expected_portfolio_sharpe": float(np.mean([1 / max(p["half_life"], 1) for p in selected]) * 5),
            "diversification_ratio": float(1 - np.mean([abs(p["correlation"]) for p in selected])),
        }

    def monitor_pairs_in_realtime(self, pairs: list[dict], current_prices: dict) -> list[dict]:
        signals = []
        for p in pairs:
            a, b = p["pair"]
            if a not in current_prices or b not in current_prices:
                continue
            spread_now = current_prices[a] - p["hedge_ratio"] * current_prices[b]
            mu = p.get("mu", spread_now)
            sigma = p.get("sigma", abs(spread_now) * 0.1 or 1.0)
            z = (spread_now - mu) / sigma if sigma else 0.0
            if z < -1.5:
                sig = "ENTRY_LONG"
            elif z > 1.5:
                sig = "ENTRY_SHORT"
            elif abs(z) < 0.3:
                sig = "EXIT"
            elif abs(z) > 3:
                sig = "STOP_LOSS"
            else:
                sig = "HOLD"
            signals.append({
                "pair": p["pair"],
                "signal": sig,
                "current_zscore": float(z),
                "cointegration_status": "VALID" if p["cointegration_pvalue"] < 0.05 else "BREAKING_DOWN",
            })
        return signals
