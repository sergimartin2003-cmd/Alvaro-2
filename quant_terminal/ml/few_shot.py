"""Few-shot / meta-learning (PROMPT 2.7).

MAML y adaptación rápida con import perezoso. Utilidades puras (numpy):
muestreo de episodios support/query y score de transferibilidad entre mercados.
"""

from __future__ import annotations

import numpy as np


class FewShotLearner:
    @staticmethod
    def sample_episode(X, y, n_support: int = 10, seed: int | None = None) -> dict:
        """Divide (X, y) en support/query para un episodio few-shot."""
        rng = np.random.default_rng(seed)
        X, y = np.asarray(X), np.asarray(y)
        n = len(X)
        idx = rng.permutation(n)
        s = min(n_support, n - 1)
        return {
            "support_X": X[idx[:s]], "support_y": y[idx[:s]],
            "query_X": X[idx[s:]], "query_y": y[idx[s:]],
        }

    @staticmethod
    def transferability_score(source_features: np.ndarray, target_features: np.ndarray) -> float:
        """Similitud coseno entre los centroides de dos dominios (0-1)."""
        a = np.asarray(source_features, dtype=float).mean(axis=0)
        b = np.asarray(target_features, dtype=float).mean(axis=0)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float((a @ b) / denom)

    def build_maml_model(self, input_dim: int, output_dim: int = 1):
        import torch  # type: ignore

        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, output_dim),
        )

    def fast_adaptation(self, model, support_set: dict, query_set: dict, n_steps: int = 5) -> dict:
        import torch  # type: ignore

        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss()
        sx = torch.tensor(support_set["X"], dtype=torch.float32)
        sy = torch.tensor(support_set["y"], dtype=torch.float32).reshape(-1, 1)
        for _ in range(n_steps):
            opt.zero_grad()
            loss = loss_fn(model(sx), sy)
            loss.backward()
            opt.step()
        return {"adapted_model": model, "final_support_loss": float(loss.item())}

    def transfer_learning_across_markets(self, source_features, target_features, model=None) -> dict:
        score = self.transferability_score(source_features, target_features)
        return {
            "transferred_model": model,
            "transferability_score": score,
            "recommendation": "transferir" if score > 0.6 else "reentrenar desde cero",
        }
