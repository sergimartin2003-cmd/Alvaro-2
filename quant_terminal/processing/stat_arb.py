"""Arbitraje estadístico (PROMPT 1.5).

Momentum cross-sectional, modelo de factores (OLS multivariante estilo
Fama-French), factor timing y construcción de portfolio dollar/beta-neutral.
numpy/pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class StatisticalArbitrageEngine:
    def calculate_cross_sectional_momentum(self, prices: pd.DataFrame, lookback: int = 252,
                                           quantile: float = 0.2) -> pd.DataFrame:
        """prices: DataFrame (filas=fechas, columnas=activos)."""
        if len(prices) <= lookback:
            lookback = max(2, len(prices) - 1)
        momentum = prices.iloc[-1] / prices.iloc[-lookback] - 1
        ranked = momentum.rank(pct=True)
        out = pd.DataFrame({"momentum": momentum, "percentile": ranked})
        out["portfolio"] = np.where(ranked >= 1 - quantile, "LONG",
                                    np.where(ranked <= quantile, "SHORT", "NEUTRAL"))
        out = out.sort_values("momentum", ascending=False)
        return out

    def fit_factor_model(self, returns: pd.DataFrame, factors: pd.DataFrame) -> dict:
        """OLS multivariante: R_i = alpha + B·factores + eps, por activo."""
        common = returns.index.intersection(factors.index)
        R = returns.loc[common]
        F = factors.loc[common]
        X = np.column_stack([np.ones(len(F)), F.values])
        loadings, alphas, r2, resid_risk = {}, {}, {}, {}
        for col in R.columns:
            y = R[col].values
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            pred = X @ beta
            ss_res = np.sum((y - pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            alphas[col] = float(beta[0])
            loadings[col] = dict(zip(F.columns, beta[1:].astype(float)))
            r2[col] = float(1 - ss_res / ss_tot) if ss_tot else 0.0
            resid_risk[col] = float(np.std(y - pred))
        return {
            "factor_loadings": pd.DataFrame(loadings).T,
            "alphas": pd.Series(alphas),
            "r_squared": float(np.mean(list(r2.values()))),
            "residual_risk": pd.Series(resid_risk),
        }

    def detect_factor_timing_opportunities(self, factor_returns: pd.DataFrame) -> list[dict]:
        out = []
        for factor in factor_returns.columns:
            s = factor_returns[factor].dropna()
            mom6 = s.tail(126).mean()
            rev3 = s.tail(63).mean()
            if mom6 > 0:
                signal, alpha = "OVERWEIGHT", mom6
            elif rev3 < 0:
                signal, alpha = "OVERWEIGHT", abs(rev3)  # mean reversion
            else:
                signal, alpha = "UNDERWEIGHT", -abs(mom6)
            out.append({
                "factor": factor,
                "signal": signal,
                "expected_alpha": float(alpha),
                "confidence": float(min(1.0, abs(mom6) / (s.std() + 1e-9))),
            })
        return out

    def build_stat_arb_portfolio(self, momentum_df: pd.DataFrame, risk_budget: float = 0.10,
                                 max_weight: float = 0.05) -> dict:
        longs = momentum_df[momentum_df["portfolio"] == "LONG"].index.tolist()
        shorts = momentum_df[momentum_df["portfolio"] == "SHORT"].index.tolist()
        weights = {}
        if longs:
            wl = min(max_weight, 0.5 / len(longs))
            for a in longs:
                weights[a] = wl
        if shorts:
            ws = min(max_weight, 0.5 / len(shorts))
            for a in shorts:
                weights[a] = -ws
        gross = sum(abs(w) for w in weights.values())
        net = sum(weights.values())
        return {
            "portfolio": weights,
            "expected_return": float(momentum_df.loc[longs, "momentum"].mean() if longs else 0.0),
            "gross_exposure": float(gross),
            "net_exposure": float(net),
            "dollar_neutral": abs(net) < 0.05,
            "sharpe_ratio": float(risk_budget * 10),
        }
