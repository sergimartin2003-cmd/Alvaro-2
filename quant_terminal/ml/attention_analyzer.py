"""Análisis de mecanismos de atención (PROMPT 2.8).

Extracción de pesos de atención de un modelo (perezoso) + utilidades puras:
softmax, normalización temporal/feature e identificación de patrones clave.
"""

from __future__ import annotations

import numpy as np


class AttentionMechanismAnalyzer:
    @staticmethod
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x = x - x.max(axis=axis, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=axis, keepdims=True)

    def summarize_attention(self, attention_weights: np.ndarray) -> dict:
        """attention_weights: matriz (queries, keys). Devuelve atención agregada."""
        w = np.asarray(attention_weights, dtype=float)
        temporal = w.mean(axis=0)  # atención media por posición/tiempo
        feature = w.mean(axis=1)
        return {
            "attention_weights": w,
            "temporal_attention": temporal,
            "feature_attention": feature,
            "entropy": float(-np.sum(self.softmax(temporal) * np.log(self.softmax(temporal) + 1e-12))),
        }

    def identify_key_patterns(self, attention_weights: np.ndarray, threshold: float = 0.1,
                              top_k: int = 5) -> list[dict]:
        summary = self.summarize_attention(attention_weights)
        temporal = summary["temporal_attention"]
        norm = temporal / (temporal.sum() + 1e-12)
        idx = np.argsort(norm)[::-1][:top_k]
        out = []
        for i in idx:
            if norm[i] >= threshold:
                out.append({
                    "position": int(i),
                    "attention_score": float(norm[i]),
                    "trading_implication": f"El modelo pondera fuertemente la posición {int(i)}",
                })
        return out

    def extract_attention_weights(self, model, input_data) -> dict:
        raise NotImplementedError(
            "Depende de la arquitectura; extrae los tensores de atención del modelo "
            "y pásalos a summarize_attention/identify_key_patterns."
        )
