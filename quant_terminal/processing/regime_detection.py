"""Detección de regímenes de mercado (PROMPT 1.8).

HMM (hmmlearn opcional, fallback por cuantiles de volatilidad), change points
(ruptures opcional, fallback por varianza móvil), clasificación de régimen y
sugerencia de estrategia. numpy/pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class RegimeDetectionEngine:
    def fit_hidden_markov_model(self, returns: pd.Series, n_states: int = 3) -> dict:
        r = pd.Series(returns).dropna()
        try:
            from hmmlearn.hmm import GaussianHMM

            X = r.values.reshape(-1, 1)
            model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100)
            model.fit(X)
            states = model.predict(X)
            current = int(states[-1])
            trans = model.transmat_
            means = model.means_.flatten()
        except Exception:
            # Fallback: estados por terciles de volatilidad rolling.
            vol = r.rolling(20, min_periods=1).std()
            q = vol.quantile([1 / n_states * i for i in range(1, n_states)])
            states = np.digitize(vol.values, q.values)
            current = int(states[-1])
            n = n_states
            trans = np.full((n, n), 1 / n)
            means = np.array([r[states == s].mean() if (states == s).any() else 0 for s in range(n)])

        # Caracterizar estados por retorno medio.
        order = np.argsort(means)
        labels = {}
        names = ["BEAR", "NEUTRAL", "BULL"] if n_states == 3 else [f"STATE_{i}" for i in range(n_states)]
        for rank, s in enumerate(order):
            labels[int(s)] = names[min(rank, len(names) - 1)]
        diag = trans[current, current] if current < len(trans) else 0.5
        expected_duration = 1 / (1 - diag) if diag < 1 else float("inf")
        return {
            "current_regime": current,
            "current_regime_label": labels.get(current, "NEUTRAL"),
            "transition_matrix": np.asarray(trans),
            "regime_characteristics": {int(s): {"mean_return": float(m)} for s, m in enumerate(means)},
            "expected_duration": float(expected_duration),
        }

    def detect_change_points(self, series: pd.Series, method: str = "variance", penalty: float = 3.0) -> list[int]:
        s = pd.Series(series).dropna().values
        try:
            import ruptures as rpt

            algo = rpt.Pelt(model="rbf").fit(s)
            return [int(c) for c in algo.predict(pen=penalty)[:-1]]
        except Exception:
            # Fallback: saltos en varianza rolling > umbral.
            vol = pd.Series(s).rolling(20, min_periods=1).std()
            dv = vol.diff().abs()
            thr = dv.mean() + penalty * dv.std()
            return [int(i) for i in np.where(dv > thr)[0]]

    def classify_current_regime(self, market_data: dict) -> dict:
        vol = market_data.get("volatility", 0.15)
        trend = market_data.get("trend_slope", 0.0)
        pcr = market_data.get("put_call_ratio", 1.0)

        if vol > 0.40:
            regime, risk = "CRISIS", "EXTREME"
        elif trend > 0.001 and vol < 0.20:
            regime, risk = "STRONG_BULL", "LOW"
        elif trend > 0:
            regime, risk = "WEAK_BULL", "MEDIUM"
        elif trend < -0.001 and vol > 0.25:
            regime, risk = "STRONG_BEAR", "HIGH"
        elif trend < 0:
            regime, risk = "WEAK_BEAR", "MEDIUM"
        else:
            regime, risk = "NEUTRAL", "MEDIUM"

        confidence = float(np.clip(abs(trend) * 200 + (0.3 if vol < 0.2 else 0), 0, 1))
        return {
            "regime": regime,
            "confidence": confidence,
            "characteristics": {"volatility": vol, "trend": trend, "put_call_ratio": pcr},
            "recommended_strategy": self.suggest_strategy_for_regime(regime)["strategy"],
            "risk_level": risk,
        }

    def suggest_strategy_for_regime(self, regime: str) -> dict:
        table = {
            "STRONG_BULL": ("momentum / trend following", {"equities": 0.8, "cash": 0.1, "bonds": 0.1}),
            "WEAK_BULL": ("selective long / quality", {"equities": 0.6, "bonds": 0.3, "cash": 0.1}),
            "NEUTRAL": ("mean reversion / market neutral", {"equities": 0.4, "bonds": 0.4, "cash": 0.2}),
            "WEAK_BEAR": ("defensive / low beta", {"equities": 0.3, "bonds": 0.5, "cash": 0.2}),
            "STRONG_BEAR": ("hedged / short bias", {"equities": 0.2, "bonds": 0.5, "cash": 0.3}),
            "CRISIS": ("cash / safe havens", {"cash": 0.6, "bonds": 0.3, "gold": 0.1}),
        }
        strat, alloc = table.get(regime, table["NEUTRAL"])
        return {
            "strategy": strat,
            "asset_allocation": alloc,
            "position_sizing": "reducida" if regime in ("STRONG_BEAR", "CRISIS") else "normal",
            "risk_management": "stops ajustados + hedging" if "BEAR" in regime or regime == "CRISIS" else "stops normales",
        }
