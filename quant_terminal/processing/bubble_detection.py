"""Detección de burbujas (PROMPT 1.11).

LPPL (Sornette) vía curve_fit, crecimiento exponencial, herding (dispersión
cross-section, CSAD) y detección en tiempo real. numpy/scipy/pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _lppl(t, A, B, tc, m, C, omega, phi):
    dt = np.clip(tc - t, 1e-6, None)
    return A + B * dt**m + C * dt**m * np.cos(omega * np.log(dt) + phi)


class BubbleDetectionEngine:
    def fit_lppl_model(self, prices: pd.Series, window: int = 60) -> dict:
        s = pd.Series(prices).dropna().tail(window)
        y = np.log(s.values)
        t = np.arange(len(y), dtype=float)
        n = len(t)
        try:
            from scipy.optimize import curve_fit

            p0 = [y[-1], -1.0, n * 1.2, 0.5, 0.0, 6.0, 0.0]
            bounds = (
                [-np.inf, -np.inf, n, 0.1, -np.inf, 1.0, -2 * np.pi],
                [np.inf, np.inf, n * 3, 0.9, np.inf, 20.0, 2 * np.pi],
            )
            popt, _ = curve_fit(_lppl, t, y, p0=p0, bounds=bounds, maxfev=5000)
            pred = _lppl(t, *popt)
            ss_res = np.sum((y - pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
            tc = popt[2]
            crash_in = tc - n
        except Exception:
            r2, tc, crash_in = 0.0, float("nan"), float("nan")

        bubble_prob = float(np.clip(r2, 0, 1)) if np.isfinite(r2) else 0.0
        return {
            "tc_index": float(tc) if np.isfinite(tc) else None,
            "confidence": float(max(r2, 0)),
            "bubble_probability": bubble_prob,
            "expected_crash_in_periods": float(crash_in) if np.isfinite(crash_in) else None,
            "current_phase": "LATE_BUBBLE" if bubble_prob > 0.8 else "EARLY" if bubble_prob > 0.5 else "NORMAL",
        }

    def detect_exponential_growth(self, prices: pd.Series, window: int = 30) -> dict:
        from scipy.stats import linregress

        s = pd.Series(prices).dropna().tail(window)
        y = np.log(s.values)
        t = np.arange(len(y))
        reg = linregress(t, y)
        doubling = np.log(2) / reg.slope if reg.slope > 0 else float("inf")
        return {
            "is_exponential": bool(reg.rvalue**2 > 0.9 and reg.slope > 0),
            "growth_rate": float(reg.slope),
            "doubling_time": float(doubling),
            "r_squared": float(reg.rvalue**2),
            "sustainability_score": float(np.clip(1 - reg.rvalue**2, 0, 1)),
        }

    def detect_herding_behavior(self, returns: pd.DataFrame, window: int = 20) -> dict:
        """CSAD: dispersión cross-section baja y anómala => herding."""
        market = returns.mean(axis=1)
        dispersion = returns.sub(market, axis=0).abs().mean(axis=1)
        recent = dispersion.tail(window)
        pctile = float((dispersion < recent.iloc[-1]).mean())
        interp = "HERDING" if pctile < 0.1 else "PANIC" if pctile > 0.9 else "NORMAL"
        return {
            "herding_score": float(1 - pctile),
            "dispersion": float(recent.iloc[-1]),
            "dispersion_percentile": pctile,
            "interpretation": interp,
        }

    def detect_bubbles_in_realtime(self, price_data: dict) -> list[dict]:
        """price_data: {symbol: pd.Series de precios}."""
        out = []
        for symbol, prices in price_data.items():
            lppl = self.fit_lppl_model(prices)
            exp = self.detect_exponential_growth(prices)
            score = 0.6 * lppl["bubble_probability"] + 0.4 * (1 if exp["is_exponential"] else 0)
            if score > 0.5:
                out.append({
                    "symbol": symbol,
                    "bubble_score": float(score),
                    "expected_crash_in_periods": lppl["expected_crash_in_periods"],
                    "confidence": lppl["confidence"],
                    "recommended_action": "SHORT" if score > 0.8 else "HEDGE" if score > 0.65 else "AVOID",
                })
        out.sort(key=lambda x: x["bubble_score"], reverse=True)
        return out
