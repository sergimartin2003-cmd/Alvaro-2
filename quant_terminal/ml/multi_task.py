"""Multi-task learning (PROMPT 2.14).

Predice retorno, volatilidad y dirección simultáneamente. Incluye una baseline
multi-task lineal (numpy, testeable) además del modelo profundo (TF, lazy).
"""

from __future__ import annotations

import numpy as np


class MultiTaskLearner:
    def __init__(self, tasks: list[str] | None = None) -> None:
        self.tasks = tasks or ["return", "volatility", "direction"]
        self.model = None
        self._W = None  # baseline lineal compartida

    # ---------------------------------------------------- baseline numpy
    def fit_linear_baseline(self, X: np.ndarray, targets: dict) -> dict:
        """Backbone lineal compartido con una cabeza por tarea (mínimos cuadrados)."""
        X = np.asarray(X, dtype=float)
        Xb = np.column_stack([np.ones(len(X)), X])
        self._W = {}
        losses = {}
        for task in self.tasks:
            y = np.asarray(targets[task], dtype=float)
            beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
            self._W[task] = beta
            losses[task] = float(np.mean((Xb @ beta - y) ** 2))
        return {"task_losses": losses}

    def predict_all_tasks(self, X: np.ndarray) -> dict:
        if self._W is None:
            raise ValueError("Ajusta la baseline o el modelo primero.")
        Xb = np.column_stack([np.ones(len(np.atleast_2d(X))), np.atleast_2d(X)])
        out = {}
        for task in self.tasks:
            pred = Xb @ self._W[task]
            if task == "direction":
                out["direction_prediction"] = np.where(pred > 0, "UP", "DOWN")
            else:
                out[f"{task}_prediction"] = pred
        return out

    @staticmethod
    def weighted_loss(task_losses: dict, task_weights: dict) -> float:
        total_w = sum(task_weights.values()) or 1.0
        return float(sum(task_losses[t] * task_weights.get(t, 0) for t in task_losses) / total_w)

    # ---------------------------------------------------- modelo profundo (lazy)
    def build_multi_task_model(self, input_dim: int):
        from tensorflow.keras import layers, models  # type: ignore

        inp = layers.Input(shape=(input_dim,))
        shared = layers.Dense(64, activation="relu")(inp)
        shared = layers.Dense(32, activation="relu")(shared)
        outs = {
            "return": layers.Dense(1, name="return")(shared),
            "volatility": layers.Dense(1, activation="softplus", name="volatility")(shared),
            "direction": layers.Dense(1, activation="sigmoid", name="direction")(shared),
        }
        self.model = models.Model(inp, [outs[t] for t in self.tasks if t in outs])
        return self.model
