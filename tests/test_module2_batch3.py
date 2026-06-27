import numpy as np

from quant_terminal.ml.contrastive_learning import ContrastiveLearningModel
from quant_terminal.ml.continual_learning import ContinualLearningModel
from quant_terminal.ml.meta_optimizer import MetaLearningOptimizer
from quant_terminal.ml.multi_task import MultiTaskLearner
from quant_terminal.ml.self_supervised import SelfSupervisedPredictor


# ---- 2.11 Contrastive ----
def test_cosine_similarity_and_find_similar():
    m = ContrastiveLearningModel()
    emb = {"A": [1.0, 0.0], "B": [0.99, 0.01], "C": [0.0, 1.0]}
    similar = m.find_similar_assets(emb, "A", top_k=2)
    assert similar[0]["symbol"] == "B"  # B es casi idéntico a A
    assert similar[0]["similarity_score"] > similar[1]["similarity_score"]


def test_info_nce_lower_when_aligned():
    m = ContrastiveLearningModel(temperature=0.1)
    rng = np.random.default_rng(0)
    anchors = rng.normal(0, 1, (8, 4))
    aligned = m.info_nce_loss(anchors, anchors.copy())  # positivos idénticos
    misaligned = m.info_nce_loss(anchors, rng.normal(0, 1, (8, 4)))
    assert aligned < misaligned


# ---- 2.12 Meta optimizer ----
def test_hyperparameter_tuning_finds_optimum():
    # objetivo: maximizar -(lr-0.01)^2 => óptimo en lr≈0.01
    opt = MetaLearningOptimizer()
    res = opt.fast_hyperparameter_tuning(
        lambda p: -((p["lr"] - 0.01) ** 2), {"lr": (0.0, 0.1)}, n_trials=200, seed=0
    )
    assert abs(res["best_hyperparameters"]["lr"] - 0.01) < 0.02
    assert res["n_trials"] == 200


def test_grid_search():
    opt = MetaLearningOptimizer()
    res = opt.fast_hyperparameter_tuning(
        lambda p: p["a"] + p["b"], {"a": [1, 2], "b": [10, 20]}, method="grid", maximize=True
    )
    assert res["best_hyperparameters"] == {"a": 2, "b": 20}


# ---- 2.13 Self-supervised ----
def test_masked_input():
    seq = np.arange(20, dtype=float)
    res = SelfSupervisedPredictor(mask_ratio=0.25).create_masked_input(seq, seed=0)
    assert res["mask"].sum() == 5  # 25% de 20
    assert np.all(res["masked_input"][res["mask"]] == 0.0)
    assert len(res["targets"]) == 5


# ---- 2.14 Multi-task ----
def test_multi_task_baseline():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (200, 3))
    targets = {
        "return": X[:, 0] * 0.5 + rng.normal(0, 0.01, 200),
        "volatility": np.abs(X[:, 1]) * 0.3 + 0.1,
        "direction": (X[:, 0] > 0).astype(float),
    }
    learner = MultiTaskLearner()
    res = learner.fit_linear_baseline(X, targets)
    assert set(res["task_losses"]) == {"return", "volatility", "direction"}
    preds = learner.predict_all_tasks(X[:5])
    assert "return_prediction" in preds
    assert set(np.unique(preds["direction_prediction"])).issubset({"UP", "DOWN"})
    wl = learner.weighted_loss(res["task_losses"], {"return": 1, "volatility": 1, "direction": 1})
    assert wl >= 0


# ---- 2.15 Continual learning ----
def test_ewc_penalty():
    cl = ContinualLearningModel(ewc_lambda=1.0)
    old = {"w": np.array([1.0, 2.0])}
    fisher = {"w": np.array([1.0, 1.0])}
    # mismos params => penalización 0
    assert cl.ewc_penalty(old, old, fisher) == 0.0
    # params desplazados => penalización > 0
    moved = {"w": np.array([2.0, 2.0])}
    assert cl.ewc_penalty(moved, old, fisher) > 0


def test_learn_new_task_consolidates():
    cl = ContinualLearningModel()
    grads = {"w": np.array([[0.1, 0.2], [0.3, 0.4]])}
    res = cl.learn_new_task({"w": np.array([1.0, 1.0])}, grads)
    assert res["consolidated"] is True
    assert cl.fisher is not None
