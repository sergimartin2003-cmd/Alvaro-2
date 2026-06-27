"""Modelos estocásticos: Ornstein-Uhlenbeck (mean reversion) y GARCH(1,1)."""

from __future__ import annotations

import numpy as np


class OrnsteinUhlenbeckModel:
    """Mean reversion: dX = theta*(mu - X)dt + sigma dW.

    Ajuste por OLS sobre dP = a + b*P, con theta = -b, mu = -a/b.
    """

    def __init__(self, prices) -> None:
        self.prices = np.asarray(prices, dtype=float)
        if self.prices.size < 3:
            raise ValueError("Se requieren al menos 3 precios")
        self.theta = self.mu = self.sigma = None

    def fit(self) -> dict:
        returns = np.diff(self.prices)
        prices_lag = self.prices[:-1]
        X = np.vstack([np.ones(len(prices_lag)), prices_lag]).T
        beta = np.linalg.lstsq(X, returns, rcond=None)[0]
        self.theta = -beta[1]
        self.mu = -beta[0] / beta[1] if beta[1] != 0 else float(self.prices.mean())
        residuals = returns - (beta[0] + beta[1] * prices_lag)
        self.sigma = float(np.std(residuals))
        half_life = np.log(2) / self.theta if self.theta > 0 else np.inf
        return {
            "theta": float(self.theta),
            "mu": float(self.mu),
            "sigma": self.sigma,
            "half_life": float(half_life),
        }

    def predict_optimal_entry(self, current_price: float):
        if self.theta is None:
            self.fit()
        denom = self.sigma / np.sqrt(2 * self.theta) if self.theta > 0 else self.sigma
        z = (current_price - self.mu) / denom if denom else 0.0
        if z < -1.5:
            return "BUY", float(z)
        if z > 1.5:
            return "SELL", float(z)
        return "HOLD", float(z)

    def simulate(self, T: float, dt: float, n_simulations: int = 1000, seed=None):
        if self.theta is None:
            self.fit()
        rng = np.random.default_rng(seed)
        n_steps = int(T / dt)
        sims = np.zeros((n_simulations, n_steps))
        for i in range(n_simulations):
            X = self.prices[-1]
            for j in range(n_steps):
                dW = rng.normal(0, np.sqrt(dt))
                X = X + self.theta * (self.mu - X) * dt + self.sigma * dW
                sims[i, j] = X
        return sims


class GARCHVolatilityModel:
    """GARCH(1,1) para volatilidad condicional. Requiere el paquete ``arch``."""

    def __init__(self, returns) -> None:
        self.returns = np.asarray(returns, dtype=float) * 100
        self.model = None
        self.results = None

    def fit_garch11(self):
        try:
            from arch import arch_model
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Instala 'arch' para GARCH: pip install arch") from exc
        self.model = arch_model(self.returns, vol="Garch", p=1, q=1)
        self.results = self.model.fit(disp="off")
        return self.results

    def forecast_volatility(self, horizon: int = 5):
        if self.results is None:
            self.fit_garch11()
        forecast = self.results.forecast(horizon=horizon)
        variance = forecast.variance.values[-1]
        return np.sqrt(variance) / 100

    def calculate_var(self, confidence: float = 0.95, horizon: int = 1) -> float:
        from scipy.stats import norm

        vol = self.forecast_volatility(horizon)[0]
        z = norm.ppf(1 - confidence)
        return float(z * vol * np.sqrt(horizon))

    def detect_volatility_regime(self):
        if self.results is None:
            self.fit_garch11()
        cond_vol = self.results.conditional_volatility
        p = np.percentile(cond_vol[~np.isnan(cond_vol)], [25, 50, 75, 90])
        current = float(cond_vol[-1])
        if current > p[3]:
            return "HIGH_VOLATILITY", current
        if current < p[0]:
            return "LOW_VOLATILITY", current
        return "NORMAL_VOLATILITY", current
