"""Normalizing Flows para distribuciones complejas (PROMPT 2.10).

El flow se construye con import perezoso (nflows/torch). Como núcleo testeable
se incluye un estimador de densidad gaussiano multivariante (numpy) para
detección de anomalías y generación de escenarios.
"""

from __future__ import annotations

import numpy as np


class NormalizingFlowsModel:
    def __init__(self) -> None:
        self.model = None
        self._mean = None
        self._cov = None

    def fit_gaussian(self, data: np.ndarray) -> dict:
        """Ajusta una gaussiana multivariante (fallback de densidad)."""
        X = np.asarray(data, dtype=float)
        self._mean = X.mean(axis=0)
        self._cov = np.cov(X, rowvar=False) + 1e-6 * np.eye(X.shape[1])
        return {"mean": self._mean, "cov": self._cov}

    def estimate_density(self, data: np.ndarray) -> np.ndarray:
        """log-likelihood por observación (baja densidad = anomalía)."""
        if self._mean is None:
            self.fit_gaussian(data)
        X = np.asarray(data, dtype=float)
        d = X.shape[1]
        diff = X - self._mean
        inv = np.linalg.inv(self._cov)
        maha = np.einsum("ij,jk,ik->i", diff, inv, diff)
        logdet = np.linalg.slogdet(self._cov)[1]
        return -0.5 * (maha + logdet + d * np.log(2 * np.pi))

    def generate_scenarios(self, n_samples: int = 1000, seed: int | None = None) -> np.ndarray:
        if self._mean is None:
            raise ValueError("Ajusta el modelo primero (fit_gaussian).")
        rng = np.random.default_rng(seed)
        return rng.multivariate_normal(self._mean, self._cov, size=n_samples)

    def build_normalizing_flow(self, input_dim: int, n_transforms: int = 8):
        from nflows.distributions.normal import StandardNormal  # type: ignore
        from nflows.flows.base import Flow  # type: ignore
        from nflows.transforms.base import CompositeTransform  # type: ignore
        from nflows.transforms.permutations import ReversePermutation  # type: ignore
        from nflows.transforms.autoregressive import (  # type: ignore
            MaskedAffineAutoregressiveTransform,
        )

        transforms = []
        for _ in range(n_transforms):
            transforms.append(ReversePermutation(features=input_dim))
            transforms.append(MaskedAffineAutoregressiveTransform(features=input_dim, hidden_features=64))
        self.model = Flow(CompositeTransform(transforms), StandardNormal([input_dim]))
        return self.model
