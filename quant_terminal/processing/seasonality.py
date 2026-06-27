"""Patrones estacionales y anomalías de calendario."""

from __future__ import annotations

import numpy as np
import pandas as pd


class SeasonalityAnalyzer:
    """Estacionalidad mensual, día de la semana y turn-of-the-month.

    ``historical_data`` es un DataFrame con índice temporal y una columna por
    activo (precios). Se recomienda >= 10 años de datos.
    """

    def __init__(self, historical_data: pd.DataFrame) -> None:
        self.data = historical_data

    @staticmethod
    def t_test(a, b) -> float:
        from scipy.stats import ttest_ind

        if len(a) < 2 or len(b) < 2:
            return float("nan")
        return float(ttest_ind(a, b, equal_var=False).pvalue)

    def calculate_monthly_seasonality(self, asset: str = "SPY") -> pd.DataFrame:
        s = self.data[asset].copy()
        df = pd.DataFrame({"price": s})
        df["month"] = df.index.month
        df["return"] = df["price"].pct_change()
        out = df.groupby("month")["return"].agg(["mean", "std", "count"])
        out["win_rate"] = df.groupby("month")["return"].apply(lambda x: (x > 0).mean())
        return out

    def calculate_day_of_week_effect(self, asset: str = "SPY") -> pd.DataFrame:
        s = self.data[asset].copy()
        df = pd.DataFrame({"price": s})
        df["dow"] = df.index.dayofweek
        df["return"] = df["price"].pct_change()
        out = df.groupby("dow")["return"].agg(["mean", "std", "count"])
        out["win_rate"] = df.groupby("dow")["return"].apply(lambda x: (x > 0).mean())
        return out

    def detect_turn_of_month(self, asset: str = "SPY") -> dict:
        s = self.data[asset].copy()
        ret = s.pct_change()
        tom, non_tom = [], []
        for date, r in ret.items():
            if np.isnan(r):
                continue
            (tom if (date.day >= 28 or date.day <= 3) else non_tom).append(r)
        tom_avg, non_avg = np.mean(tom), np.mean(non_tom)
        return {
            "turn_of_month_avg": float(tom_avg),
            "non_turn_of_month_avg": float(non_avg),
            "effect_strength": float(tom_avg - non_avg),
            "statistical_significance": self.t_test(tom, non_tom),
        }

    def generate_seasonal_signal(self, asset: str = "SPY") -> list:
        now = pd.Timestamp.now()
        monthly = self.calculate_monthly_seasonality(asset)
        dow = self.calculate_day_of_week_effect(asset)
        tom = self.detect_turn_of_month(asset)
        signals = []

        if now.month in monthly.index:
            m_ret = monthly.loc[now.month, "mean"]
            m_win = monthly.loc[now.month, "win_rate"]
            if m_ret > 0.01 and m_win > 0.6:
                signals.append(("MONTHLY", "BULLISH", m_ret, m_win))
            elif m_ret < -0.01 and m_win < 0.4:
                signals.append(("MONTHLY", "BEARISH", m_ret, m_win))

        if (now.day >= 28 or now.day <= 3) and tom["effect_strength"] > 0.002:
            signals.append(("TURN_OF_MONTH", "BULLISH", tom["effect_strength"]))

        if now.dayofweek in dow.index:
            d_ret = dow.loc[now.dayofweek, "mean"]
            if d_ret > 0.002:
                signals.append(("DAY_OF_WEEK", "BULLISH", d_ret))
            elif d_ret < -0.002:
                signals.append(("DAY_OF_WEEK", "BEARISH", d_ret))
        return signals
