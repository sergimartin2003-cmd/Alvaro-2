"""Redes neuronales bayesianas (PROMPT 2.5).

Construcción con MC Dropout (TF, perezoso). La agregación de incertidumbre a
partir de muestras MC y la detección de out-of-distribution son numpy puro
(testeable).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class BayesianNeuralNetwork:
    def __init__(self, dropout_rate: float = 0.2) -> None:
        self.dropout_rate = dropout_rate
        self.model = None

    def build_bayesian_nn(self, input_dim: int, output_dim: int = 1):
        from tensorflow.keras import layers, models  # type: ignore

        inp = layers.Input(shape=(input_dim,))
        x = layers.Dense(64, activation="relu")(inp)
        x = layers.Dropout(self.dropout_rate)(x, training=True)  # MC dropout
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dropout(self.dropout_rate)(x, training=True)
        out = layers.Dense(output_dim)(x)
        self.model = models.Model(inp, out)
        self.model.compile(optimizer="adam", loss="mse")
        return self.model

    @staticmethod
    def aggregate_uncertainty(samples: np.ndarray, ci: float = 0.95) -> dict:
        """Media, std e intervalos a partir de muestras MC (n_samples, n_obs)."""
        samples = np.asarray(samples, dtype=float)
        z = 1.959963984540054 if abs(ci - 0.95) < 1e-9 else 2.575829303548901
        mean = samples.mean(axis=0)
        std = samples.std(axis=0)
        return {
            "predictions": mean,
            "uncertainty": std,
            "lower": mean - z * std,
            "upper": mean + z * std,
        }

    def predict_with_uncertainty(self, model, input_data, n_samples: int = 100, ci: float = 0.95) -> dict:
        X = np.asarray(input_data, dtype=float)
        preds = np.array([np.asarray(model(X, training=True)).reshape(-1) for _ in range(n_samples)])
        agg = self.aggregate_uncertainty(preds, ci)
        low_conf = agg["uncertainty"] > np.percentile(agg["uncertainty"], 90)
        return {**agg, "low_confidence_mask": low_conf}

    @staticmethod
    def detect_out_of_distribution(uncertainty: np.ndarray, threshold_pct: float = 95) -> np.ndarray:
        """Marca como OOD las observaciones con incertidumbre por encima del percentil."""
        u = np.asarray(uncertainty, dtype=float)
        thr = np.percentile(u, threshold_pct)
        return (u > thr).astype(int)
