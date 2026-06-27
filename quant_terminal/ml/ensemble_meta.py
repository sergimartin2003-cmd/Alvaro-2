"""Ensemble con meta-learning (PROMPT 2.3).

Stacking, blending y ponderación dinámica. La ponderación dinámica y la
combinación de predicciones son numpy puro (testeable). El stacking acepta
modelos con API fit/predict (sklearn-compatibles).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class EnsembleMetaLearner:
    def __init__(self) -> None:
        self.base_models = []
        self.meta_model = None
        self.weights = None

    def build_stacking_ensemble(self, base_models: list, meta_model, X, y) -> dict:
        """Entrena base models, usa sus predicciones como features del meta-model."""
        meta_features = []
        for m in base_models:
            m.fit(X, y)
            meta_features.append(np.asarray(m.predict(X)).reshape(-1))
        meta_X = np.column_stack(meta_features)
        meta_model.fit(meta_X, y)
        self.base_models = base_models
        self.meta_model = meta_model
        pred = meta_model.predict(meta_X)
        mse = float(np.mean((np.asarray(y) - pred) ** 2))
        return {"ensemble": {"base": base_models, "meta": meta_model},
                "meta_features_shape": meta_X.shape, "train_mse": mse}

    @staticmethod
    def dynamic_weighting(performance: pd.DataFrame, lookback: int = 30,
                          metric: str = "sharpe") -> dict:
        """Pesos por performance reciente (Sharpe del retorno de cada modelo).

        performance: DataFrame (filas=tiempo, columnas=modelos) con retornos.
        """
        recent = performance.tail(lookback)
        if metric == "sharpe":
            score = recent.mean() / (recent.std() + 1e-9)
        else:
            score = recent.mean()
        score = score.clip(lower=0)
        total = score.sum()
        if total <= 0:
            n = performance.shape[1]
            return {c: 1.0 / n for c in performance.columns}
        return {c: float(score[c] / total) for c in performance.columns}

    def combine_predictions(self, predictions: dict, weights: dict) -> np.ndarray:
        """Combina predicciones por pesos (numpy)."""
        keys = list(predictions)
        stacked = np.vstack([np.asarray(predictions[k], dtype=float) for k in keys])
        w = np.array([weights.get(k, 0.0) for k in keys])
        w = w / (w.sum() + 1e-12)
        return (stacked * w[:, None]).sum(axis=0)

    def online_learning_update(self, weights: dict, recent_performance: pd.DataFrame,
                               alpha: float = 0.3) -> dict:
        """Mezcla los pesos actuales con los nuevos (EMA) para estabilidad."""
        new = self.dynamic_weighting(recent_performance)
        return {k: float((1 - alpha) * weights.get(k, 0) + alpha * new.get(k, 0)) for k in new}
