import numpy as np
import pandas as pd

from quant_terminal.ml.attention_analyzer import AttentionMechanismAnalyzer
from quant_terminal.ml.few_shot import FewShotLearner
from quant_terminal.ml.graph_nn import GraphNeuralNetwork
from quant_terminal.ml.neural_ode import NeuralODEModel
from quant_terminal.ml.normalizing_flows import NormalizingFlowsModel


def _returns(seed=0, n=200, k=5):
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 0.01, n)
    return pd.DataFrame({f"A{i}": common * (0.5 + 0.2 * i) + rng.normal(0, 0.003, n) for i in range(k)})


# ---- 2.6 Graph NN ----
def test_graph_and_spillover():
    eng = GraphNeuralNetwork()
    graph = eng.build_asset_graph(_returns(), correlation_threshold=0.1)
    assert len(graph["assets"]) == 5
    res = eng.predict_spillover_effects(graph, "A0", 1.0)
    # El origen del shock conserva el mayor impacto (incluye feedback del grafo).
    assert res["spillover_predictions"]["A0"] >= 1.0
    assert any(abs(v) > 0 for k, v in res["spillover_predictions"].items() if k != "A0")
    assert len(res["most_affected_assets"]) >= 1
    assert res["total_impact"] > 0


# ---- 2.7 Few-shot ----
def test_sample_episode_and_transfer():
    X = np.arange(40).reshape(20, 2)
    y = np.arange(20)
    ep = FewShotLearner.sample_episode(X, y, n_support=8, seed=0)
    assert len(ep["support_X"]) == 8
    assert len(ep["query_X"]) == 12
    score = FewShotLearner.transferability_score(np.ones((10, 3)), np.ones((10, 3)))
    assert abs(score - 1.0) < 1e-6


# ---- 2.8 Attention ----
def test_attention_softmax_and_patterns():
    a = AttentionMechanismAnalyzer()
    sm = a.softmax(np.array([1.0, 2.0, 3.0]))
    assert abs(sm.sum() - 1.0) < 1e-9
    weights = np.array([[0.1, 0.8, 0.1], [0.2, 0.7, 0.1]])
    patterns = a.identify_key_patterns(weights, threshold=0.1, top_k=3)
    assert patterns[0]["position"] == 1  # la posición 1 domina


# ---- 2.9 Neural ODE ----
def test_rk4_exponential_decay():
    # dy/dt = -y => y(t) = y0 * e^{-t}
    t = np.linspace(0, 2, 50)
    y = NeuralODEModel.rk4_integrate(lambda y, _t: -y, 1.0, t)
    assert abs(y[-1] - np.exp(-2)) < 1e-3


def test_fit_and_predict_trajectory():
    rng = np.random.default_rng(1)
    # serie mean-reverting alrededor de 100
    s = [100.0]
    for _ in range(200):
        s.append(s[-1] + 0.3 * (100 - s[-1]) + rng.normal(0, 1))
    ode = NeuralODEModel()
    dyn = ode.fit_linear_dynamics(s)
    assert dyn["mean_reversion"] is True
    traj = ode.predict_continuous_trajectory(80.0, np.arange(20))
    assert np.all(traj["lower"] <= traj["upper"])


# ---- 2.10 Normalizing flows (Gaussian fallback) ----
def test_density_and_generation():
    rng = np.random.default_rng(2)
    data = rng.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], size=500)
    nf = NormalizingFlowsModel()
    nf.fit_gaussian(data)
    ll = nf.estimate_density(data)
    # un punto lejano debe tener menor log-likelihood que el centro
    center_ll = nf.estimate_density(np.array([[0, 0]]))[0]
    far_ll = nf.estimate_density(np.array([[10, 10]]))[0]
    assert center_ll > far_ll
    samples = nf.generate_scenarios(100, seed=0)
    assert samples.shape == (100, 2)
