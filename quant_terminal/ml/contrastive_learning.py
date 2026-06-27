"""Contrastive learning para representaciones de mercado (PROMPT 2.11).

El encoder se construye con import perezoso (TF). Núcleo testeable (numpy):
loss InfoNCE, matriz de similitud coseno y búsqueda de activos similares.
"""

from __future__ import annotations

import numpy as np


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norm + 1e-12)


class ContrastiveLearningModel:
    def __init__(self, embedding_dim: int = 128, temperature: float = 0.1) -> None:
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.model = None

    @staticmethod
    def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
        X = _l2_normalize(np.asarray(embeddings, dtype=float))
        return X @ X.T

    def info_nce_loss(self, anchors: np.ndarray, positives: np.ndarray) -> float:
        """InfoNCE: cada anchor empareja con su positivo; resto son negativos."""
        a = _l2_normalize(np.asarray(anchors, dtype=float))
        p = _l2_normalize(np.asarray(positives, dtype=float))
        logits = (a @ p.T) / self.temperature
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        diag = np.clip(np.diag(probs), 1e-12, 1.0)
        return float(-np.mean(np.log(diag)))

    def find_similar_assets(self, embeddings: dict, query_asset: str, top_k: int = 10) -> list[dict]:
        names = list(embeddings)
        mat = np.array([embeddings[n] for n in names], dtype=float)
        sim = self.cosine_similarity_matrix(mat)
        qi = names.index(query_asset)
        order = np.argsort(sim[qi])[::-1]
        out = []
        for j in order:
            if names[j] == query_asset:
                continue
            out.append({"symbol": names[j], "similarity_score": float(sim[qi, j])})
            if len(out) >= top_k:
                break
        return out

    def learn_market_embeddings(self, model, market_data) -> dict:
        emb = np.asarray(model.predict(np.asarray(market_data), verbose=0))
        return {"embeddings": emb, "similarity_matrix": self.cosine_similarity_matrix(emb)}

    def build_contrastive_model(self, input_dim: int):
        from tensorflow.keras import layers, models  # type: ignore

        self.model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(256, activation="relu"),
            layers.Dense(self.embedding_dim),
            layers.Lambda(lambda x: x / (1e-12 + __import__("tensorflow").norm(x, axis=1, keepdims=True))),
        ])
        return self.model
