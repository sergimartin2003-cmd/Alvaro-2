"""Análisis cross-asset: correlaciones rolling, cambios de régimen, risk appetite."""

from __future__ import annotations

import numpy as np
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


class CrossAssetCorrelationAnalyzer:
    """Correlaciones dinámicas (PROMPT 1.7): DCC (EWMA), lead-lag, rupturas,
    spillover (Diebold-Yilmaz simplificado). numpy/pandas."""

    def calculate_dcc_correlations(self, returns: pd.DataFrame, lam: float = 0.94) -> dict:
        """DCC aproximado vía correlación condicional EWMA."""
        r = returns.dropna()
        ewm_corr = r.ewm(alpha=1 - lam).corr()
        last = ewm_corr.loc[ewm_corr.index.get_level_values(0)[-1]]
        # Historial de correlación media entre pares.
        cols = list(r.columns)
        hist = pd.Series(index=r.index, dtype=float)
        rolling = r.rolling(20).corr()
        for ts in r.index[20:]:
            try:
                m = rolling.loc[ts].values
                iu = np.triu_indices_from(m, k=1)
                hist[ts] = np.nanmean(m[iu])
            except Exception:
                continue
        return {
            "correlation_matrix": last.values,
            "assets": cols,
            "correlation_history": hist.dropna(),
            "correlation_volatility": float(hist.dropna().std()),
            "correlation_forecast": last.values,
        }

    def detect_leading_lagging_relationships(self, prices: pd.DataFrame, max_lag: int = 10) -> list[dict]:
        rets = prices.pct_change().dropna()
        out = []
        cols = list(rets.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                best_lag, best_corr = 0, 0.0
                for lag in range(-max_lag, max_lag + 1):
                    if lag == 0:
                        continue
                    c = rets[a].corr(rets[b].shift(lag))
                    if c is not None and abs(c) > abs(best_corr):
                        best_corr, best_lag = c, lag
                if abs(best_corr) > 0.2 and best_lag != 0:
                    leader, lagger = (a, b) if best_lag > 0 else (b, a)
                    out.append({
                        "leader": leader,
                        "lagger": lagger,
                        "optimal_lag": abs(best_lag),
                        "correlation_at_lag": float(best_corr),
                        "trading_implication": f"{leader} anticipa a {lagger} en {abs(best_lag)} períodos",
                    })
        return out

    def detect_correlation_breakdowns(self, correlation_history: pd.Series, threshold: float = 2.0) -> list[dict]:
        s = correlation_history.dropna()
        if len(s) < 10:
            return []
        change = s.diff(20)
        z = (change - change.mean()) / (change.std() + 1e-9)
        out = []
        for ts, zval in z.dropna().items():
            if abs(zval) > threshold:
                out.append({
                    "timestamp": ts,
                    "magnitude": float(change.loc[ts]),
                    "z_score": float(zval),
                    "market_regime": "STRESS" if zval > 0 else "DECOUPLING",
                })
        return out

    def calculate_spillover_index(self, returns: pd.DataFrame, window: int = 20) -> dict:
        """Spillover simplificado a partir de R^2 cruzados de VAR(1)."""
        r = returns.dropna()
        cols = list(r.columns)
        n = len(cols)
        spill = np.zeros((n, n))
        for i, target in enumerate(cols):
            y = r[target].iloc[1:].values
            for j, src in enumerate(cols):
                x = r[src].iloc[:-1].values
                if np.std(x) < 1e-12:
                    continue
                beta = np.polyfit(x, y, 1)[0]
                pred = beta * x
                ss_tot = np.sum((y - y.mean()) ** 2)
                contrib = 1 - np.sum((y - pred) ** 2) / ss_tot if ss_tot else 0
                spill[i, j] = max(contrib, 0)
        row_sum = spill.sum(axis=1, keepdims=True)
        norm_spill = np.divide(spill, row_sum, out=np.zeros_like(spill), where=row_sum != 0)
        off_diag = norm_spill.sum() - np.trace(norm_spill)
        total = off_diag / n * 100 if n else 0.0
        net = {cols[j]: float(norm_spill[:, j].sum() - norm_spill[j, :].sum()) for j in range(n)}
        return {
            "total_spillover": float(total),
            "net_spillovers": net,
            "spillover_matrix": norm_spill,
            "systemic_risk_level": "HIGH" if total > 50 else "MEDIUM" if total > 25 else "LOW",
        }
