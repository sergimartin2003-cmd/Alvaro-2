"""Neural ODEs para series temporales (PROMPT 2.9).

El modelo neural ODE se construye con import perezoso (torchdiffeq). Se incluye
un integrador RK4 (numpy) y predicción de trayectoria continua con una dinámica
lineal ajustada por datos, ambos testeables sin libs externas.
"""

from __future__ import annotations

import numpy as np


class NeuralODEModel:
    def __init__(self, hidden_dim: int = 64) -> None:
        self.hidden_dim = hidden_dim
        self.model = None
        self._drift = None  # dinámica lineal ajustada para el fallback

    @staticmethod
    def rk4_integrate(f, y0: float, t: np.ndarray) -> np.ndarray:
        """Integrador Runge-Kutta 4 genérico (numpy)."""
        y = np.zeros(len(t))
        y[0] = y0
        for i in range(len(t) - 1):
            h = t[i + 1] - t[i]
            k1 = f(y[i], t[i])
            k2 = f(y[i] + h / 2 * k1, t[i] + h / 2)
            k3 = f(y[i] + h / 2 * k2, t[i] + h / 2)
            k4 = f(y[i] + h * k3, t[i] + h)
            y[i + 1] = y[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return y

    def fit_linear_dynamics(self, series) -> dict:
        """Ajusta dy/dt ≈ a + b·y (mean-reversion lineal)."""
        s = np.asarray(series, dtype=float)
        dy = np.diff(s)
        y = s[:-1]
        b, a = np.polyfit(y, dy, 1)
        self._drift = (a, b)
        return {"a": float(a), "b": float(b), "mean_reversion": bool(b < 0)}

    def predict_continuous_trajectory(self, initial_state: float, time_points: np.ndarray) -> dict:
        if self._drift is None:
            raise ValueError("Llama a fit_linear_dynamics primero (o usa el modelo entrenado).")
        a, b = self._drift

        def f(y, _t):
            return a + b * y

        traj = self.rk4_integrate(f, initial_state, np.asarray(time_points, dtype=float))
        # Banda de confianza simple proporcional a la distancia temporal.
        spread = 0.05 * np.abs(traj) * np.sqrt(np.arange(1, len(traj) + 1))
        return {"trajectory": traj, "lower": traj - spread, "upper": traj + spread}

    def build_neural_ode(self, input_dim: int):
        import torch  # type: ignore
        from torchdiffeq import odeint  # type: ignore  # noqa: F401

        class ODEFunc(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, 64), torch.nn.Tanh(),
                    torch.nn.Linear(64, input_dim),
                )

            def forward(self, t, y):
                return self.net(y)

        self.model = ODEFunc()
        return self.model
