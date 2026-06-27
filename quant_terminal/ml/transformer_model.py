"""Transformers para series temporales (PROMPT 2.1).

TFT / PatchTST construidos con imports perezosos (transformers/torch/keras).
Incluye utilidades puras (numpy) para preparar patches y derivar intervalos de
confianza a partir de muestras MC, que sí son testeables sin libs pesadas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class TransformerTimeSeriesModel:
    def __init__(self, seq_len: int = 96, pred_len: int = 24) -> None:
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.model = None

    # --------------------------------------------------------- utilidades puras
    @staticmethod
    def make_patches(series, patch_len: int = 16, stride: int = 8) -> np.ndarray:
        """Divide una serie 1D en patches solapados (numpy)."""
        x = np.asarray(series, dtype=float)
        patches = [x[i : i + patch_len] for i in range(0, len(x) - patch_len + 1, stride)]
        return np.array(patches) if patches else np.empty((0, patch_len))

    @staticmethod
    def confidence_intervals(samples: np.ndarray, lower: float = 5, upper: float = 95) -> dict:
        """Intervalos de confianza a partir de muestras (n_samples, horizon)."""
        samples = np.asarray(samples, dtype=float)
        return {
            "mean": samples.mean(axis=0),
            "lower": np.percentile(samples, lower, axis=0),
            "upper": np.percentile(samples, upper, axis=0),
            "std": samples.std(axis=0),
        }

    # ----------------------------------------------------------- modelos (lazy)
    def build_temporal_fusion_transformer(self, config: dict):
        from transformers import (  # type: ignore
            TimeSeriesTransformerConfig,
            TimeSeriesTransformerForPrediction,
        )

        cfg = TimeSeriesTransformerConfig(
            prediction_length=config.get("pred_len", self.pred_len),
            context_length=config.get("seq_len", self.seq_len),
            num_attention_heads=config.get("num_heads", 8),
            num_static_real_features=config.get("n_static", 0),
        )
        self.model = TimeSeriesTransformerForPrediction(cfg)
        return self.model

    def train_patchtst(self, data: pd.DataFrame, patch_len: int = 16, stride: int = 8):
        raise NotImplementedError(
            "PatchTST requiere torch + datos etiquetados; usa make_patches para la "
            "preparación e integra tu loop de entrenamiento."
        )

    def generate_predictions(self, model, input_data: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
        raise NotImplementedError(
            "Requiere un modelo entrenado; usa confidence_intervals sobre las "
            "muestras del modelo para los intervalos."
        )
