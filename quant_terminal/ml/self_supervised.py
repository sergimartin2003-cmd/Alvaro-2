"""Modelos self-supervised: masked prediction (PROMPT 2.13).

Pre-entrenamiento por enmascarado tipo BERT (modelo lazy en TF). El núcleo
testeable (numpy) genera la entrada enmascarada y la máscara objetivo.
"""

from __future__ import annotations

import numpy as np


class SelfSupervisedPredictor:
    def __init__(self, mask_ratio: float = 0.15) -> None:
        self.mask_ratio = mask_ratio
        self.model = None

    def create_masked_input(self, sequence: np.ndarray, mask_value: float = 0.0,
                            seed: int | None = None) -> dict:
        """Enmascara una fracción de los time steps; devuelve entrada y máscara."""
        x = np.asarray(sequence, dtype=float).copy()
        rng = np.random.default_rng(seed)
        n = x.shape[0]
        k = max(1, int(round(self.mask_ratio * n)))
        idx = rng.choice(n, size=k, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        masked = x.copy()
        masked[mask] = mask_value
        return {"masked_input": masked, "mask": mask, "targets": x[mask], "target_indices": idx}

    def build_masked_prediction_model(self, input_dim: int, seq_len: int):
        from tensorflow.keras import layers, models  # type: ignore

        self.model = models.Sequential([
            layers.Input(shape=(seq_len, input_dim)),
            layers.LSTM(64, return_sequences=True),
            layers.Dense(input_dim),
        ])
        self.model.compile(optimizer="adam", loss="mse")
        return self.model

    def pretrain_on_unlabeled_data(self, model, unlabeled_data, epochs: int = 100):
        raise NotImplementedError(
            "Integra create_masked_input en tu loop: enmascara, predice y minimiza "
            "MSE solo sobre las posiciones enmascaradas."
        )

    def fine_tune_on_labeled_data(self, model, labeled_data, epochs: int = 10):
        raise NotImplementedError("Carga los pesos pre-entrenados y ajusta con cabeza supervisada.")
