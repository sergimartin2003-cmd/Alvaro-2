import numpy as np
import pandas as pd

from quant_terminal.ml.bayesian_nn import BayesianNeuralNetwork
from quant_terminal.ml.ensemble_meta import EnsembleMetaLearner
from quant_terminal.ml.gan_market import GenerativeAdversarialNetwork
from quant_terminal.ml.rl_trader import ReinforcementLearningTrader, TradingEnvironment
from quant_terminal.ml.transformer_model import TransformerTimeSeriesModel


# ---- 2.1 Transformer utils ----
def test_make_patches():
    patches = TransformerTimeSeriesModel.make_patches(np.arange(100), patch_len=16, stride=8)
    assert patches.shape[1] == 16
    assert patches.shape[0] > 0


def test_confidence_intervals():
    rng = np.random.default_rng(0)
    samples = rng.normal(5, 1, (200, 3))
    ci = TransformerTimeSeriesModel.confidence_intervals(samples)
    assert np.all(ci["lower"] <= ci["mean"])
    assert np.all(ci["mean"] <= ci["upper"])


# ---- 2.2 RL trader ----
def test_trading_environment_step():
    prices = pd.Series(100 + np.cumsum(np.random.default_rng(1).normal(0, 1, 100)))
    env = TradingEnvironment(prices, window=5)
    state = env.reset()
    assert len(state) == 6  # window + position
    state, reward, done, info = env.step(1)  # LONG
    assert "equity" in info
    assert isinstance(reward, float)


def test_backtest_buy_and_hold_policy():
    prices = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(2).normal(0.001, 0.01, 200))))
    trader = ReinforcementLearningTrader()
    res = trader.backtest_rl_strategy(lambda s: 1, prices, window=5)  # siempre LONG
    assert res["final_equity"] > 0
    assert "sharpe_ratio" in res
    assert res["n_steps"] > 0


# ---- 2.3 Ensemble meta ----
def test_dynamic_weighting():
    rng = np.random.default_rng(3)
    perf = pd.DataFrame({
        "good": rng.normal(0.01, 0.01, 60),
        "bad": rng.normal(-0.01, 0.02, 60),
    })
    w = EnsembleMetaLearner.dynamic_weighting(perf)
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert w["good"] > w["bad"]


def test_combine_predictions():
    eng = EnsembleMetaLearner()
    preds = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}
    combined = eng.combine_predictions(preds, {"a": 0.5, "b": 0.5})
    assert np.allclose(combined, [2.0, 3.0])


def test_stacking_with_sklearn():
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(4)
    X = rng.normal(0, 1, (100, 3))
    y = X[:, 0] * 2 + rng.normal(0, 0.1, 100)
    res = EnsembleMetaLearner().build_stacking_ensemble(
        [LinearRegression(), LinearRegression()], LinearRegression(), X, y
    )
    assert res["train_mse"] < 1.0


# ---- 2.4 GAN validation ----
def test_validate_synthetic_data():
    rng = np.random.default_rng(5)
    real = pd.DataFrame(rng.normal(0, 1, (100, 2)))
    synth = pd.DataFrame(rng.normal(0, 1, (100, 2)))
    res = GenerativeAdversarialNetwork.validate_synthetic_data(real, synth)
    assert "mean_abs_divergence" in res
    assert res["quality"] in ("GOOD", "POOR")


# ---- 2.5 Bayesian NN ----
def test_aggregate_uncertainty_and_ood():
    rng = np.random.default_rng(6)
    samples = rng.normal(0, 1, (100, 5))
    agg = BayesianNeuralNetwork.aggregate_uncertainty(samples)
    assert np.all(agg["lower"] <= agg["upper"])
    ood = BayesianNeuralNetwork.detect_out_of_distribution(agg["uncertainty"], threshold_pct=80)
    assert set(np.unique(ood)).issubset({0, 1})
