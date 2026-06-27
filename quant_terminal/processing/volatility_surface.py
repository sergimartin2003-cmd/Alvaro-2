"""Análisis de superficie de volatilidad implícita (PROMPT 1.1).

Calcula IV vía Newton-Raphson sobre Black-Scholes, construye la superficie,
mide smile/skew/term structure y detecta arbitraje de volatilidad y VRP.
Solo depende de numpy/scipy/pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


class VolatilitySurfaceAnalyzer:
    def __init__(self, risk_free_rate: float = 0.05) -> None:
        self.r = risk_free_rate
        self._cache: dict = {}

    # ------------------------------------------------------------ Black-Scholes
    def black_scholes_price(self, S, K, T, r, sigma, option_type="call") -> float:
        if T <= 0 or sigma <= 0:
            intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
            return float(intrinsic)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "call":
            return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))

    def black_scholes_vega(self, S, K, T, r, sigma) -> float:
        if T <= 0 or sigma <= 0:
            return 0.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return float(S * norm.pdf(d1) * np.sqrt(T))

    def calculate_implied_volatility(self, option_price, strike, underlying_price,
                                     days_to_expiry, option_type="call") -> float:
        """Newton-Raphson robusto con bisección de respaldo."""
        T = max(days_to_expiry / 365.0, 1e-6)
        S, K = underlying_price, strike
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        if option_price <= intrinsic + 1e-6:
            return float("nan")
        sigma = 0.3
        for _ in range(100):
            price = self.black_scholes_price(S, K, T, self.r, sigma, option_type)
            vega = self.black_scholes_vega(S, K, T, self.r, sigma)
            diff = price - option_price
            if abs(diff) < 1e-5:
                return float(sigma)
            if vega < 1e-8:
                break
            sigma = sigma - diff / vega
            if sigma <= 0 or sigma > 5:
                sigma = 0.3
                break
        # Respaldo: bisección.
        lo, hi = 1e-3, 5.0
        for _ in range(100):
            mid = (lo + hi) / 2
            price = self.black_scholes_price(S, K, T, self.r, mid, option_type)
            if abs(price - option_price) < 1e-5:
                return float(mid)
            if price > option_price:
                hi = mid
            else:
                lo = mid
        return float((lo + hi) / 2)

    # --------------------------------------------------------------- superficie
    def build_volatility_surface(self, options_chain: pd.DataFrame) -> dict:
        df = options_chain.copy()
        df["mid"] = (df["bid"] + df["ask"]) / 2
        # Filtrar ilíquidas (spread relativo > 10%).
        df = df[(df["ask"] - df["bid"]) <= 0.10 * df["mid"].clip(lower=1e-6)]
        ivs = []
        for _, row in df.iterrows():
            ivs.append(
                self.calculate_implied_volatility(
                    row["mid"], row["strike"], row["underlying_price"],
                    row["days_to_expiry"], row.get("option_type", "call"),
                )
            )
        df["iv"] = ivs
        df = df.dropna(subset=["iv"])
        if df.empty:
            return {"surface_quality": 0.0, "atm_iv": float("nan")}

        strikes = np.sort(df["strike"].unique())
        expiries = np.sort(df["days_to_expiry"].unique())
        spot = float(df["underlying_price"].iloc[0])

        atm_idx = (df["strike"] - spot).abs().idxmin()
        atm_iv = float(df.loc[atm_idx, "iv"])

        smile = {int(e): df[df["days_to_expiry"] == e].set_index("strike")["iv"].to_dict()
                 for e in expiries}
        term = {float(k): df[df["strike"] == k].set_index("days_to_expiry")["iv"].to_dict()
                for k in strikes}

        skew_index = self._skew_index(df, spot)
        return {
            "strikes": strikes,
            "expiries": expiries,
            "smile_by_expiry": smile,
            "term_structure_by_strike": term,
            "skew_index": skew_index,
            "atm_iv": atm_iv,
            "surface_quality": float(len(df) / max(len(options_chain), 1)),
        }

    @staticmethod
    def _skew_index(df: pd.DataFrame, spot: float) -> float:
        """Proxy de skew 25-delta: IV(OTM put) - IV(OTM call)."""
        otm_put = df[(df.get("option_type") == "put") & (df["strike"] < spot)]
        otm_call = df[(df.get("option_type") == "call") & (df["strike"] > spot)]
        if otm_put.empty or otm_call.empty:
            return 0.0
        return float(otm_put["iv"].mean() - otm_call["iv"].mean())

    # -------------------------------------------------------------- VRP / arb
    def calculate_volatility_risk_premium(self, symbol, options_chain, historical_prices,
                                          window: int = 30) -> dict:
        surface = self.build_volatility_surface(options_chain)
        iv = surface.get("atm_iv", float("nan"))
        prices = pd.Series(historical_prices)
        rv = float(np.log(prices / prices.shift(1)).dropna().tail(window).std() * np.sqrt(252))
        vrp = iv - rv if np.isfinite(iv) else float("nan")
        signal = "NEUTRAL"
        if np.isfinite(vrp):
            if vrp > 0.05:
                signal = "SELL_VOL"
            elif vrp < -0.05:
                signal = "BUY_VOL"
        return {
            "vrp_current": vrp,
            "iv_atm": iv,
            "rv_30d": rv,
            "signal": signal,
            "confidence": float(min(1.0, abs(vrp) / 0.1)) if np.isfinite(vrp) else 0.0,
        }

    def detect_volatility_arbitrage(self, surface: dict) -> list[dict]:
        opps = []
        smile = surface.get("smile_by_expiry", {})
        # Calendar: IV(corto) > IV(largo) (backwardation de vol).
        expiries = sorted(smile)
        if len(expiries) >= 2:
            atm = surface.get("atm_iv")
            iv_short = np.mean(list(smile[expiries[0]].values()))
            iv_long = np.mean(list(smile[expiries[-1]].values()))
            if iv_short > iv_long + 0.02:
                opps.append({
                    "type": "CALENDAR", "expected_profit": float(iv_short - iv_long),
                    "confidence": 0.6, "risk": 0.4,
                    "rationale": "IV corto > IV largo: vender corto / comprar largo",
                })
        # Butterfly sobre el smile más cercano.
        if expiries:
            near = smile[expiries[0]]
            ks = sorted(near)
            if len(ks) >= 3:
                lo, mid, hi = near[ks[0]], near[ks[len(ks) // 2]], near[ks[-1]]
                if lo + hi < 2 * mid - 0.02:
                    opps.append({
                        "type": "BUTTERFLY", "expected_profit": float(2 * mid - lo - hi),
                        "confidence": 0.5, "risk": 0.5, "rationale": "Smile cóncavo anómalo",
                    })
        return opps

    def detect_skew_anomalies(self, surface: dict, historical_skew=None) -> list[dict]:
        skew = surface.get("skew_index", 0.0)
        out = []
        if historical_skew is not None and len(historical_skew) > 5:
            arr = np.asarray(historical_skew, dtype=float)
            z = (skew - arr.mean()) / (arr.std() + 1e-9)
            if abs(z) > 2:
                out.append({
                    "type": "EXTREME_SKEW", "current_skew": skew, "z_score": float(z),
                    "severity": float(min(1.0, abs(z) / 4)),
                    "interpretation": "Skew extremo vs histórico",
                    "trading_implication": "Posible mean reversion del skew (risk reversal)",
                })
        if skew < 0:
            out.append({
                "type": "INVERTED_SKEW", "current_skew": skew, "z_score": 0.0,
                "severity": 0.5, "interpretation": "Calls más caras que puts",
                "trading_implication": "Posible euforia/squeeze al alza",
            })
        return out
