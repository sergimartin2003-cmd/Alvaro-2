"""Continual learning con Elastic Weight Consolidation (PROMPT 2.15).

Evita catastrophic forgetting penalizando cambios en parámetros importantes
para tareas previas. El cálculo de la penalización EWC y la importancia de
Fisher son numpy puro (testeable). El modelo profundo es lazy (TF/torch).
"""

from __future__ import annotations

import numpy as np


class ContinualLearningModel:
    def __init__(self, ewc_lambda: float = 0.4) -> None:
        self.ewc_lambda = ewc_lambda
        self.model = None
        self.old_params: dict | None = None
        self.fisher: dict | None = None

    @staticmethod
    def estimate_fisher(gradients: dict) -> dict:
        """Importancia de Fisher ≈ media del cuadrado de los gradientes."""
        return {k: np.mean(np.asarray(g, dtype=float) ** 2, axis=0)
                if np.asarray(g).ndim > 1 else float(np.mean(np.asarray(g) ** 2))
                for k, g in gradients.items()}

    def ewc_penalty(self, params: dict, old_params: dict, fisher: dict) -> float:
        """Penalización EWC = lambda/2 * sum F_i * (theta_i - theta_i*)^2."""
        total = 0.0
        for k in params:
            if k in old_params and k in fisher:
                diff = np.asarray(params[k], dtype=float) - np.asarray(old_params[k], dtype=float)
                total += float(np.sum(np.asarray(fisher[k], dtype=float) * diff**2))
        return 0.5 * self.ewc_lambda * total

    def consolidate(self, params: dict, gradients: dict) -> None:
        """Guarda parámetros e importancia tras aprender una tarea."""
        self.old_params = {k: np.asarray(v, dtype=float).copy() for k, v in params.items()}
        self.fisher = self.estimate_fisher(gradients)

    def learn_new_task(self, params: dict, gradients: dict) -> dict:
        penalty = 0.0
        if self.old_params is not None and self.fisher is not None:
            penalty = self.ewc_penalty(params, self.old_params, self.fisher)
        self.consolidate(params, gradients)
        return {"ewc_penalty": penalty, "consolidated": True}

    def build_continual_learning_model(self, input_dim: int, output_dim: int = 1):
        from tensorflow.keras import layers, models  # type: ignore

        self.model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(output_dim),
        ])
        self.model.compile(optimizer="adam", loss="mse")
        return self.model
