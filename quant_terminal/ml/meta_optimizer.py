"""Meta-learning para optimización de hiperparámetros (PROMPT 2.12).

Búsqueda (random/grid) sobre un espacio de hiperparámetros guiada por una
función objetivo. Núcleo numpy puro, testeable sin libs externas.
"""

from __future__ import annotations

import itertools

import numpy as np


class MetaLearningOptimizer:
    def __init__(self, base_optimizer: str = "adam") -> None:
        self.base_optimizer = base_optimizer
        self.history: list[dict] = []

    def build_meta_optimizer(self) -> dict:
        return {"base_optimizer": self.base_optimizer, "history": self.history}

    def fast_hyperparameter_tuning(self, objective, search_space: dict, n_trials: int = 10,
                                   method: str = "random", maximize: bool = True,
                                   seed: int | None = None) -> dict:
        """objective: dict de hiperparámetros -> score (float)."""
        rng = np.random.default_rng(seed)
        candidates = self._candidates(search_space, n_trials, method, rng)
        results = []
        for params in candidates:
            score = float(objective(params))
            results.append({"params": params, "score": score})
            self.history.append(results[-1])
        best = (max if maximize else min)(results, key=lambda r: r["score"])
        return {
            "best_hyperparameters": best["params"],
            "best_score": best["score"],
            "n_trials": len(results),
            "all_results": results,
        }

    @staticmethod
    def _candidates(search_space: dict, n_trials: int, method: str, rng) -> list[dict]:
        keys = list(search_space)
        if method == "grid":
            combos = list(itertools.product(*[search_space[k] for k in keys]))
            return [dict(zip(keys, c)) for c in combos]
        # random
        out = []
        for _ in range(n_trials):
            params = {}
            for k in keys:
                vals = search_space[k]
                if isinstance(vals, tuple) and len(vals) == 2:  # rango (lo, hi)
                    params[k] = float(rng.uniform(*vals))
                else:  # lista discreta
                    params[k] = vals[int(rng.integers(len(vals)))]
            out.append(params)
        return out
