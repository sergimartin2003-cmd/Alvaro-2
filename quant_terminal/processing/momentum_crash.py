"""Detección de crashes de momentum (PROMPT 1.14).

Momentum crashes ocurren tras caídas fuertes cuando el mercado rebota: los
'losers' suben más que los 'winners'. Mide el riesgo y ajusta la exposición.
numpy/pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class MomentumCrashDetector:
    def calculate_momentum_crash_risk(self, momentum_returns: pd.Series,
                                      market_returns: pd.Series) -> dict:
        mom = pd.Series(momentum_returns).dropna()
        mkt = pd.Series(market_returns).dropna()
        idx = mom.index.intersection(mkt.index)
        mom, mkt = mom.loc[idx], mkt.loc[idx]

        # Señales de riesgo: mercado en drawdown reciente + alta volatilidad.
        bear = mkt.tail(126).sum() < 0
        recent_rebound = mkt.tail(20).sum() > 0
        mkt_vol = mkt.tail(60).std() * np.sqrt(252)
        # Momentum tiende a sufrir cuando hay bear reciente + rebote + vol alta.
        score = 0.0
        if bear:
            score += 0.4
        if bear and recent_rebound:
            score += 0.3
        if mkt_vol > 0.30:
            score += 0.3
        score = float(np.clip(score, 0, 1))

        regime = "CRASH_RISK" if score > 0.6 else "ELEVATED" if score > 0.3 else "NORMAL"
        exposure = float(np.clip(1 - score, 0.1, 1.0))
        return {
            "crash_risk_score": score,
            "current_regime": regime,
            "market_volatility": float(mkt_vol),
            "expected_momentum_return": float(mom.tail(20).mean() * (1 - score)),
            "recommended_exposure": exposure,
            "hedging_strategy": "reducir momentum / añadir value" if score > 0.5 else "mantener",
        }

    def time_momentum_factor(self, market_data: dict) -> dict:
        crash = self.calculate_momentum_crash_risk(
            pd.Series(market_data.get("momentum_returns", [0.0])),
            pd.Series(market_data.get("market_returns", [0.0])),
        )
        score = crash["crash_risk_score"]
        momentum_w = float(np.clip(0.6 - score * 0.5, 0.05, 0.6))
        value_w = float(np.clip(0.2 + score * 0.3, 0.2, 0.6))
        cash_w = float(np.clip(score * 0.4, 0.0, 0.5))
        total = momentum_w + value_w + cash_w
        return {
            "momentum_exposure": momentum_w / total,
            "value_exposure": value_w / total,
            "cash_allocation": cash_w / total,
            "expected_alpha": float(0.02 * (1 - score)),
            "confidence": float(1 - score),
        }
