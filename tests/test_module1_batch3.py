import numpy as np
import pandas as pd

from quant_terminal.processing.bubble_detection import BubbleDetectionEngine
from quant_terminal.processing.carry_trade import CarryTradeOptimizer
from quant_terminal.processing.crash_prediction import CrashPredictionEngine
from quant_terminal.processing.liquidity_shock import LiquidityShockDetector
from quant_terminal.processing.momentum_crash import MomentumCrashDetector


# ---- 1.11 Bubble detection ----
def test_exponential_growth_detected():
    t = np.arange(60)
    prices = pd.Series(10 * np.exp(0.03 * t))  # crecimiento exponencial puro
    res = BubbleDetectionEngine().detect_exponential_growth(prices)
    assert res["is_exponential"] is True
    assert res["growth_rate"] > 0


def test_lppl_and_realtime():
    rng = np.random.default_rng(0)
    t = np.arange(80)
    prices = pd.Series(100 * np.exp(0.02 * t) + rng.normal(0, 1, 80))
    eng = BubbleDetectionEngine()
    lppl = eng.fit_lppl_model(prices)
    assert 0 <= lppl["bubble_probability"] <= 1
    rt = eng.detect_bubbles_in_realtime({"X": prices})
    assert isinstance(rt, list)


def test_herding():
    rng = np.random.default_rng(1)
    # Activos muy correlacionados => baja dispersión => herding.
    common = rng.normal(0, 0.02, 100)
    rets = pd.DataFrame({f"A{i}": common + rng.normal(0, 0.001, 100) for i in range(5)})
    res = BubbleDetectionEngine().detect_herding_behavior(rets)
    assert "interpretation" in res
    assert 0 <= res["herding_score"] <= 1


# ---- 1.12 Crash prediction ----
def test_critical_slowing_down():
    rng = np.random.default_rng(2)
    calm = rng.normal(0, 0.005, 100)
    turbulent = rng.normal(0, 0.03, 100)
    series = pd.Series(np.r_[calm, turbulent])
    res = CrashPredictionEngine().detect_critical_slowing_down(series)
    assert 0 <= res["crash_probability"] <= 1


def test_early_warning_signals():
    res = CrashPredictionEngine().calculate_early_warning_signals({
        "vix_backwardation": 0.25, "credit_spread_change_1m": 0.6,
        "yield_curve_slope": -0.1, "put_call_ratio": 1.3,
    })
    assert res["overall_risk_level"] == "EXTREME"
    assert res["crash_probability"] > 0.75


def test_crash_scenarios():
    res = CrashPredictionEngine().simulate_crash_scenarios(
        {"AAPL": 0.5, "TLT": 0.5}, shock_magnitude=-0.2, betas={"AAPL": 1.5, "TLT": -0.2}
    )
    assert res["portfolio_impact"] < 0
    assert len(res["most_vulnerable_assets"]) >= 1


# ---- 1.13 Carry optimizer ----
def test_carry_opportunities_and_portfolio():
    rates = {"USD": 5.5, "JPY": 0.1, "MXN": 11.0, "EUR": 4.0}
    opt = CarryTradeOptimizer()
    opps = opt.calculate_carry_opportunities(rates)
    assert opps[0]["risk_adjusted_carry"] >= opps[-1]["risk_adjusted_carry"]
    port = opt.optimize_carry_portfolio(opps, max_positions=3)
    assert abs(sum(port["weights"].values()) - 1.0) < 1e-6
    assert port["expected_carry"] > 0


# ---- 1.14 Momentum crash ----
def test_momentum_crash_risk():
    rng = np.random.default_rng(3)
    mkt = pd.Series(np.r_[rng.normal(-0.003, 0.02, 120), rng.normal(0.01, 0.03, 20)])
    mom = pd.Series(rng.normal(0, 0.01, 140))
    res = MomentumCrashDetector().calculate_momentum_crash_risk(mom, mkt)
    assert 0 <= res["crash_risk_score"] <= 1
    assert 0 <= res["recommended_exposure"] <= 1
    timing = MomentumCrashDetector().time_momentum_factor(
        {"momentum_returns": mom.tolist(), "market_returns": mkt.tolist()}
    )
    assert abs(timing["momentum_exposure"] + timing["value_exposure"] + timing["cash_allocation"] - 1.0) < 1e-6


# ---- 1.15 Liquidity shock ----
def test_liquidity_shocks_and_score():
    det = LiquidityShockDetector()
    shocks = det.detect_liquidity_shocks({
        "spread": 0.5, "avg_spread": 0.05, "volume": 100, "avg_volume": 1000,
        "ted_spread": 1.5, "symbols": ["AAPL"],
    })
    types = {s["type"] for s in shocks}
    assert "SPREAD_BLOWOUT" in types and "FUNDING_STRESS" in types
    score = det.calculate_liquidity_score({"spread_bps": 5, "daily_volume": 1e7, "volatility": 0.15})
    assert 0 <= score["liquidity_score"] <= 100
    assert score["liquidity_grade"] in list("ABCDF")
