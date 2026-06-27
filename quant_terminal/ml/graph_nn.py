"""Graph Neural Networks para relaciones entre activos (PROMPT 2.6).

Grafo de activos por correlación (numpy) y propagación de spillover testeable.
El modelo GNN se construye con import perezoso (torch_geometric / dgl).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class GraphNeuralNetwork:
    def build_asset_graph(self, returns: pd.DataFrame, correlation_threshold: float = 0.5) -> dict:
        corr = returns.corr()
        assets = list(corr.columns)
        adj = (corr.abs() >= correlation_threshold).astype(float).values.copy()
        np.fill_diagonal(adj, 0)
        edges = [(assets[i], assets[j]) for i in range(len(assets))
                 for j in range(i + 1, len(assets)) if adj[i, j]]
        return {"assets": assets, "adjacency_matrix": adj, "edges": edges,
                "node_features": returns.tail(20).T.values}

    def build_gnn_model(self, graph: dict, hidden_dim: int = 32):
        import torch  # type: ignore
        from torch_geometric.nn import GCNConv  # type: ignore

        n_feat = graph["node_features"].shape[1]

        class GCN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.c1 = GCNConv(n_feat, hidden_dim)
                self.c2 = GCNConv(hidden_dim, 1)

            def forward(self, x, edge_index):
                x = self.c1(x, edge_index).relu()
                return self.c2(x, edge_index)

        return GCN()

    def predict_spillover_effects(self, graph: dict, shock_asset: str,
                                  shock_magnitude: float, decay: float = 0.6,
                                  steps: int = 5) -> dict:
        """Propaga un shock por el grafo (difusión normalizada, numpy)."""
        assets = graph["assets"]
        adj = np.asarray(graph["adjacency_matrix"], dtype=float)
        deg = adj.sum(axis=1, keepdims=True)
        norm_adj = np.divide(adj, deg, out=np.zeros_like(adj), where=deg != 0)

        impact = np.zeros(len(assets))
        if shock_asset in assets:
            impact[assets.index(shock_asset)] = shock_magnitude
        total = impact.copy()
        wave = impact.copy()
        for _ in range(steps):
            wave = decay * (norm_adj @ wave)
            total = total + wave
            if np.allclose(wave, 0):
                break
        predictions = {a: float(v) for a, v in zip(assets, total)}
        order = sorted(assets, key=lambda a: abs(predictions[a]), reverse=True)
        return {
            "spillover_predictions": predictions,
            "most_affected_assets": [a for a in order if a != shock_asset][:3],
            "propagation_path": order,
            "total_impact": float(np.abs(total).sum()),
        }
