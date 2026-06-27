"""GANs para datos de mercado (PROMPT 2.4).

Construcción del GAN con import perezoso (TensorFlow). Incluye validación
estadística de datos sintéticos (numpy) testeable sin TF.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class GenerativeAdversarialNetwork:
    def __init__(self, latent_dim: int = 100) -> None:
        self.latent_dim = latent_dim
        self.generator = None
        self.discriminator = None

    def build_time_series_gan(self, input_dim: int, seq_len: int = 50) -> dict:
        from tensorflow.keras import layers, models  # type: ignore

        generator = models.Sequential([
            layers.Input(shape=(self.latent_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dense(seq_len * input_dim, activation="tanh"),
            layers.Reshape((seq_len, input_dim)),
        ])
        discriminator = models.Sequential([
            layers.Input(shape=(seq_len, input_dim)),
            layers.LSTM(64),
            layers.Dense(1),
        ])
        self.generator, self.discriminator = generator, discriminator
        return {"generator": generator, "discriminator": discriminator}

    def generate_synthetic_scenarios(self, generator, n_scenarios: int = 1000) -> np.ndarray:
        noise = np.random.default_rng().normal(0, 1, (n_scenarios, self.latent_dim))
        return generator.predict(noise, verbose=0)

    @staticmethod
    def validate_synthetic_data(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict:
        """Compara momentos estadísticos clave entre real y sintético (numpy)."""
        real = np.asarray(real, dtype=float).reshape(-1)
        synth = np.asarray(synthetic, dtype=float).reshape(-1)

        def _stats(x):
            return {
                "mean": float(np.mean(x)),
                "std": float(np.std(x)),
                "skew": float(((x - x.mean()) ** 3).mean() / (x.std() ** 3 + 1e-12)),
                "kurtosis": float(((x - x.mean()) ** 4).mean() / (x.std() ** 4 + 1e-12)),
            }

        rs, ss = _stats(real), _stats(synth)
        divergence = float(np.mean([abs(rs[k] - ss[k]) for k in rs]))
        return {"real_stats": rs, "synthetic_stats": ss, "mean_abs_divergence": divergence,
                "quality": "GOOD" if divergence < 0.5 else "POOR"}
