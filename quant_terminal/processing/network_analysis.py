"""Análisis de redes financieras (PROMPT 1.10).

Red de correlaciones, centralidades, comunidades, simulación de contagio (SIR)
y riesgo sistémico (CoVaR/MES). networkx es opcional con fallback numpy para
centralidad de grado y contagio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class NetworkAnalysisEngine:
    def build_correlation_network(self, returns: pd.DataFrame, threshold: float = 0.5) -> dict:
        corr = returns.corr()
        assets = list(corr.columns)
        adj = (corr.abs() >= threshold).astype(int).values.copy()
        np.fill_diagonal(adj, 0)
        edges = [(assets[i], assets[j], float(corr.iloc[i, j]))
                 for i in range(len(assets)) for j in range(i + 1, len(assets)) if adj[i, j]]
        net = {"assets": assets, "adjacency_matrix": adj, "edges": edges, "correlation": corr}
        try:
            import networkx as nx

            g = nx.Graph()
            g.add_nodes_from(assets)
            for a, b, w in edges:
                g.add_edge(a, b, weight=abs(w))
            net["graph"] = g
        except Exception:
            net["graph"] = None
        return net

    def calculate_centrality_measures(self, network: dict) -> pd.DataFrame:
        assets = network["assets"]
        g = network.get("graph")
        if g is not None:
            import networkx as nx

            data = {
                "degree_centrality": nx.degree_centrality(g),
                "betweenness_centrality": nx.betweenness_centrality(g),
                "closeness_centrality": nx.closeness_centrality(g),
                "eigenvector_centrality": _safe_eigenvector(g),
                "pagerank": nx.pagerank(g),
            }
            return pd.DataFrame(data)
        # Fallback: solo grado normalizado.
        adj = network["adjacency_matrix"]
        deg = adj.sum(axis=1) / max(len(assets) - 1, 1)
        return pd.DataFrame({"degree_centrality": deg}, index=assets)

    def detect_communities(self, network: dict, method: str = "louvain") -> dict:
        g = network.get("graph")
        if g is None:
            return {"communities": [network["assets"]], "modularity": 0.0}
        import networkx as nx

        try:
            comms = nx.community.greedy_modularity_communities(g)
            communities = [sorted(c) for c in comms]
            mod = nx.community.modularity(g, comms)
        except Exception:
            communities, mod = [list(g.nodes())], 0.0
        return {"communities": communities, "modularity": float(mod),
                "n_communities": len(communities)}

    def simulate_contagion(self, network: dict, initial_shock: dict,
                           transmission_rate: float = 0.5, steps: int = 10) -> dict:
        """Modelo SIR sobre la matriz de adyacencia (numpy)."""
        assets = network["assets"]
        adj = np.asarray(network["adjacency_matrix"], dtype=float)
        n = len(assets)
        infected = np.array([initial_shock.get(a, 0.0) for a in assets], dtype=float)
        path = [list(np.where(infected > 0)[0])]
        order = []
        for _ in range(steps):
            spread = transmission_rate * (adj @ infected) / (adj.sum(axis=1) + 1e-9)
            new = np.clip(infected + spread, 0, 1)
            newly = np.where((new > 0.5) & (infected <= 0.5))[0]
            order.extend(int(i) for i in newly)
            if np.allclose(new, infected):
                break
            infected = new
        infected_assets = [assets[i] for i in np.where(infected > 0.5)[0]]
        return {
            "contagion_path": [assets[i] for i in order],
            "infected_assets": infected_assets,
            "total_impact": float(infected.sum() / n),
            "systemic_risk": float(len(infected_assets) / n),
            "most_vulnerable": [assets[i] for i in np.argsort(-infected)[:3]],
        }

    def calculate_systemic_risk_measures(self, network: dict, returns: pd.DataFrame) -> dict:
        """CoVaR y MES simplificados por activo."""
        assets = network["assets"]
        system = returns[assets].mean(axis=1)
        var_system = np.percentile(system, 5)
        covar, mes = {}, {}
        for a in assets:
            tail = returns[a] <= np.percentile(returns[a], 5)
            covar[a] = float(system[tail].mean()) if tail.any() else 0.0
            mes[a] = float(returns[a][system <= var_system].mean()) if (system <= var_system).any() else 0.0
        ranking = sorted(assets, key=lambda x: mes[x])
        return {
            "systemic_risk_score": float(-np.mean(list(mes.values()))),
            "covar_by_asset": covar,
            "mes_by_asset": mes,
            "most_systemically_important": ranking[:3],
        }


def _safe_eigenvector(g):
    import networkx as nx

    try:
        return nx.eigenvector_centrality(g, max_iter=1000)
    except Exception:
        return nx.degree_centrality(g)
