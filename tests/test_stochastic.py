import numpy as np

from quant_terminal.processing.stochastic_models import OrnsteinUhlenbeckModel


def test_ou_fit_recovers_mean_reversion():
    rng = np.random.default_rng(0)
    theta, mu, sigma = 0.5, 100.0, 1.0
    x, prices = mu, []
    for _ in range(2000):
        x += theta * (mu - x) * 0.1 + sigma * rng.normal(0, np.sqrt(0.1))
        prices.append(x)
    params = OrnsteinUhlenbeckModel(prices).fit()
    assert params["theta"] > 0
    assert 90 < params["mu"] < 110
    assert params["half_life"] > 0


def test_ou_entry_signal():
    prices = [100, 101, 99, 100, 102, 98, 100] * 10
    model = OrnsteinUhlenbeckModel(prices)
    model.fit()
    action, z = model.predict_optimal_entry(80)
    assert action in ("BUY", "SELL", "HOLD")
    assert isinstance(z, float)


def test_ou_simulate_shape():
    prices = list(range(100, 200))
    sims = OrnsteinUhlenbeckModel(prices).simulate(T=10, dt=1, n_simulations=5, seed=1)
    assert sims.shape == (5, 10)
