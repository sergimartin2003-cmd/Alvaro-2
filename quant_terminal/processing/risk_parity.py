"""Optimización de portfolio: Risk Parity y Hierarchical Risk Parity (HRP)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class RiskParityOptimizer:
    """Asigna pesos para igualar la contribución de riesgo de cada activo."""

    def __init__(self, returns: pd.DataFrame) -> None:
        self.returns = returns
        self.cov_matrix = returns.cov()
        self.n_assets = len(returns.columns)

    def calculate_risk_contributions(self, weights: np.ndarray) -> np.ndarray:
        cov = self.cov_matrix.values
        portfolio_vol = np.sqrt(weights @ cov @ weights)
        mcr = (cov @ weights) / portfolio_vol
        rc = weights * mcr
        return rc / portfolio_vol

    def risk_parity_objective(self, weights: np.ndarray) -> float:
        rc_pct = self.calculate_risk_contributions(weights)
        target = 1.0 / self.n_assets
        return float(np.sum((rc_pct - target) ** 2))

    def optimize(self) -> dict:
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        x0 = np.array([1.0 / self.n_assets] * self.n_assets)
        result = minimize(
            self.risk_parity_objective, x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
        )
        w = result.x
        return {
            "weights": dict(zip(self.returns.columns, w)),
            "risk_contributions": self.calculate_risk_contributions(w).tolist(),
            "portfolio_volatility": float(np.sqrt(w @ self.cov_matrix.values @ w)),
        }

    def hierarchical_risk_parity(self) -> dict:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform

        corr = self.returns.corr()
        dist = np.sqrt((1 - corr) / 2)
        link = linkage(squareform(dist.values, checks=False), method="single")
        order = leaves_list(link)
        ordered_assets = [self.returns.columns[i] for i in order]
        return self._recursive_bisection(ordered_assets)

    def _recursive_bisection(self, items: list) -> dict:
        if len(items) == 1:
            return {items[0]: 1.0}
        mid = len(items) // 2
        c1, c2 = items[:mid], items[mid:]
        v1, v2 = self._cluster_volatility(c1), self._cluster_volatility(c2)
        alpha = 1 - (v1 / (v1 + v2)) if (v1 + v2) else 0.5
        w1 = self._recursive_bisection(c1)
        w2 = self._recursive_bisection(c2)
        out = {a: w * alpha for a, w in w1.items()}
        out.update({a: w * (1 - alpha) for a, w in w2.items()})
        return out

    def _cluster_volatility(self, cluster_assets: list) -> float:
        cov = self.returns[cluster_assets].cov().values
        w = np.array([1.0 / len(cluster_assets)] * len(cluster_assets))
        return float(np.sqrt(w @ cov @ w))
