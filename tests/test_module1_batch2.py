import numpy as np
import pandas as pd

from quant_terminal.processing.anomaly_detection import AnomalyDetectionEngine
from quant_terminal.processing.cross_asset import CrossAssetCorrelationAnalyzer
from quant_terminal.processing.hf_alpha import HighFrequencyAlphaEngine
from quant_terminal.processing.network_analysis import NetworkAnalysisEngine
from quant_terminal.processing.regime_detection import RegimeDetectionEngine


def _returns(seed=0, n=300, k=4):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    base = rng.normal(0, 0.01, n)
    data = {f"A{i}": base * (0.5 + i * 0.2) + rng.normal(0, 0.005, n) for i in range(k)}
    return pd.DataFrame(data, index=idx)


# ---- 1.6 HF alpha ----
def test_hf_alpha_and_rv():
    rng = np.random.default_rng(1)
    ticks = pd.DataFrame({
        "price": 100 + np.cumsum(rng.normal(0, 0.05, 500)),
        "volume": rng.integers(1, 10, 500),
        "bid_volume": rng.integers(1, 100, 500),
        "ask_volume": rng.integers(1, 100, 500),
    })
    eng = HighFrequencyAlphaEngine()
    alpha = eng.calculate_hf_alpha_signals(ticks)
    assert len(alpha) == len(ticks)
    rv = eng.calculate_realized_volatility(ticks)
    assert (rv.dropna() >= 0).all()
    pred = eng.predict_short_term_returns(ticks[["price", "volume"]].pct_change().fillna(0))
    assert len(pred) == len(ticks)


# ---- 1.7 Cross-asset correlation ----
def test_dcc_and_spillover():
    rets = _returns()
    a = CrossAssetCorrelationAnalyzer()
    dcc = a.calculate_dcc_correlations(rets)
    assert dcc["correlation_matrix"].shape == (4, 4)
    spill = a.calculate_spillover_index(rets)
    assert 0 <= spill["total_spillover"] <= 100
    assert spill["systemic_risk_level"] in ("LOW", "MEDIUM", "HIGH")


def test_lead_lag():
    rng = np.random.default_rng(2)
    n = 300
    leader = rng.normal(0, 1, n)
    follower = np.r_[0, 0, leader[:-2]] + rng.normal(0, 0.1, n)
    prices = pd.DataFrame({"L": 100 + np.cumsum(leader), "F": 100 + np.cumsum(follower)})
    rels = CrossAssetCorrelationAnalyzer().detect_leading_lagging_relationships(prices, max_lag=5)
    assert isinstance(rels, list)


# ---- 1.8 Regime detection ----
def test_regime_classification():
    eng = RegimeDetectionEngine()
    bull = eng.classify_current_regime({"volatility": 0.12, "trend_slope": 0.003, "put_call_ratio": 0.7})
    assert bull["regime"] == "STRONG_BULL"
    crisis = eng.classify_current_regime({"volatility": 0.55, "trend_slope": -0.01})
    assert crisis["regime"] == "CRISIS"
    assert "asset_allocation" in eng.suggest_strategy_for_regime("CRISIS")


def test_hmm_fallback_and_change_points():
    rng = np.random.default_rng(5)
    rets = pd.Series(np.r_[rng.normal(0.001, 0.005, 150), rng.normal(-0.002, 0.03, 150)])
    res = RegimeDetectionEngine().fit_hidden_markov_model(rets, n_states=3)
    assert "current_regime" in res
    cps = RegimeDetectionEngine().detect_change_points(rets)
    assert isinstance(cps, list)


# ---- 1.9 Anomaly detection ----
def test_statistical_outliers():
    s = pd.Series([1, 2, 3, 2, 1, 2, 100, 2, 1])
    eng = AnomalyDetectionEngine()
    for method in ("zscore", "iqr", "mad"):
        flags = eng.detect_statistical_outliers(s, method=method, threshold=2.0)
        assert flags.iloc[6] == 1


def test_market_anomalies():
    prices = list(100 * np.exp(np.cumsum([0.0] * 10 + [-0.02] * 5)))
    res = AnomalyDetectionEngine().detect_market_anomalies({"prices": prices, "vix": 45})
    types = {a["type"] for a in res}
    assert "FLASH_CRASH" in types
    assert "VOLATILITY_SPIKE" in types


# ---- 1.10 Network analysis ----
def test_network_and_contagion():
    rets = _returns(seed=3, k=5)
    eng = NetworkAnalysisEngine()
    net = eng.build_correlation_network(rets, threshold=0.1)
    assert len(net["assets"]) == 5
    cent = eng.calculate_centrality_measures(net)
    assert "degree_centrality" in cent.columns
    contagion = eng.simulate_contagion(net, {"A0": 1.0}, transmission_rate=0.8)
    assert 0 <= contagion["systemic_risk"] <= 1
    sysrisk = eng.calculate_systemic_risk_measures(net, rets)
    assert len(sysrisk["mes_by_asset"]) == 5
