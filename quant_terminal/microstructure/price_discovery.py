"""Descubrimiento de precios (PROMPT 3.6).

Information share (Hasbrouck simplificado), relaciones lead-lag y eficiencia de
mercado (variance ratio). numpy/pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PriceDiscoveryModel:
    def calculate_information_share(self, markets: dict) -> dict:
        """markets: {nombre: serie de precios}. Share ~ proporción de varianza
        de innovación que cada mercado aporta al precio eficiente común."""
        names = list(markets)
        innov_var = {}
        for name, prices in markets.items():
            r = pd.Series(prices).pct_change().dropna()
            innov_var[name] = float(r.var())
        total = sum(innov_var.values()) or 1.0
        shares = {n: innov_var[n] / total for n in names}
        dominant = max(shares, key=shares.get)
        ranking = sorted(names, key=lambda n: shares[n], reverse=True)
        return {"information_shares": shares, "dominant_market": dominant,
                "efficiency_ranking": ranking}

    def detect_lead_lag_relationships(self, asset1: pd.Series, asset2: pd.Series,
                                      max_lag: int = 10) -> dict:
        r1 = pd.Series(asset1).pct_change().dropna()
        r2 = pd.Series(asset2).pct_change().dropna()
        idx = r1.index.intersection(r2.index)
        r1, r2 = r1.loc[idx], r2.loc[idx]
        best_lag, best_corr = 0, 0.0
        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                continue
            c = r1.corr(r2.shift(lag))
            if c is not None and abs(c) > abs(best_corr):
                best_corr, best_lag = c, lag
        leader, lagger = ("asset1", "asset2") if best_lag > 0 else ("asset2", "asset1")
        return {
            "leader": leader,
            "lagger": lagger,
            "optimal_lag": abs(best_lag),
            "correlation": float(best_corr),
            "trading_opportunity": f"{leader} anticipa a {lagger}" if abs(best_corr) > 0.3 else "débil",
        }

    def measure_market_efficiency(self, prices: pd.Series, q: int = 2) -> dict:
        """Variance ratio test: VR≈1 => random walk (eficiente)."""
        r = np.log(pd.Series(prices) / pd.Series(prices).shift(1)).dropna().values
        n = len(r)
        var_1 = np.var(r, ddof=1)
        rq = np.array([r[i : i + q].sum() for i in range(n - q + 1)])
        var_q = np.var(rq, ddof=1)
        vr = var_q / (q * var_1) if var_1 else 1.0
        return {
            "efficiency_score": float(1 - abs(vr - 1)),
            "variance_ratio": float(vr),
            "is_random_walk": bool(abs(vr - 1) < 0.2),
            "predictability": "MEAN_REVERTING" if vr < 0.8 else "TRENDING" if vr > 1.2 else "EFFICIENT",
        }
