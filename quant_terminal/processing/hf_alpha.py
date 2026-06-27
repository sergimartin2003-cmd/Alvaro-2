"""Alpha de alta frecuencia (PROMPT 1.6).

Señales de alpha (order book imbalance, trade flow), arbitraje de latencia,
realized volatility (incl. bipower) y predicción de retornos a corto plazo.
sklearn es opcional; hay fallback por mínimos cuadrados (numpy).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class HighFrequencyAlphaEngine:
    def calculate_hf_alpha_signals(self, tick_data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Combina imbalance y trade-flow correlacionados con retornos futuros."""
        df = tick_data.copy()
        ret = df["price"].pct_change().fillna(0)
        future = ret.shift(-1).fillna(0)

        if {"bid_volume", "ask_volume"}.issubset(df.columns):
            imb = (df["bid_volume"] - df["ask_volume"]) / (df["bid_volume"] + df["ask_volume"]).replace(0, np.nan)
        else:
            imb = pd.Series(0.0, index=df.index)
        imb = imb.fillna(0)

        sign = np.sign(ret).replace(0, 1)
        trade_flow = (sign * df.get("volume", pd.Series(1.0, index=df.index))).fillna(0)
        trade_flow = trade_flow / (trade_flow.abs().rolling(window, min_periods=1).mean() + 1e-9)

        alpha = (
            imb.rolling(window, min_periods=1).corr(future).fillna(0) * imb
            + 0.5 * trade_flow.rolling(window, min_periods=1).mean()
        )
        return alpha.rename("hf_alpha")

    def calculate_realized_volatility(self, trades: pd.DataFrame, window: int = 50,
                                      method: str = "simple") -> pd.Series:
        r = np.log(trades["price"] / trades["price"].shift(1)).fillna(0)
        if method == "bipower":
            bp = (np.pi / 2) * (r.abs() * r.abs().shift(1)).fillna(0)
            return bp.rolling(window, min_periods=1).sum().pow(0.5).rename("rv_bipower")
        return (r.pow(2)).rolling(window, min_periods=1).sum().pow(0.5).rename("rv")

    def detect_latency_arbitrage(self, series_a: pd.Series, series_b: pd.Series,
                                 max_lag: int = 5, corr_threshold: float = 0.5) -> list[dict]:
        """Si A se mueve y B (correlacionado) aún no, oportunidad en B."""
        ra = series_a.pct_change().fillna(0)
        rb = series_b.pct_change().fillna(0)
        out = []
        for lag in range(1, max_lag + 1):
            c = ra.corr(rb.shift(-lag))
            if c is not None and c > corr_threshold:
                last_move = ra.iloc[-1]
                if abs(last_move) > ra.std():
                    out.append({
                        "opportunity": f"B sigue a A con lag {lag}",
                        "expected_profit": float(abs(last_move) * c),
                        "required_latency": float(lag),
                        "confidence": float(c),
                    })
        return out

    def predict_short_term_returns(self, features: pd.DataFrame, target: pd.Series | None = None):
        """Ridge si sklearn está; si no, mínimos cuadrados regularizados (numpy)."""
        X = features.fillna(0).values
        if target is None:
            target = features.get("ret", pd.Series(np.zeros(len(features)))).shift(-1).fillna(0)
        y = np.asarray(target, dtype=float)
        try:
            from sklearn.linear_model import Ridge

            model = Ridge(alpha=1.0).fit(X, y)
            pred = model.predict(X)
        except Exception:
            lam = 1.0
            Xb = np.column_stack([np.ones(len(X)), X])
            A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
            beta = np.linalg.solve(A, Xb.T @ y)
            pred = Xb @ beta
        return pd.Series(pred, index=features.index, name="pred_return")
