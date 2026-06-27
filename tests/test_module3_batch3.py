import numpy as np
import pandas as pd

from quant_terminal.microstructure.liquidity_cycles import LiquidityCycleAnalyzer
from quant_terminal.microstructure.market_quality import MarketQualityMonitor
from quant_terminal.microstructure.market_resilience import MarketResilienceAnalyzer
from quant_terminal.microstructure.order_flow_predictor import OrderFlowPredictor
from quant_terminal.microstructure.toxicity_filter import ToxicityFlowFilter


def _trades(seed=0, n=400):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "price": 100 + np.cumsum(rng.normal(0, 0.05, n)),
        "volume": rng.integers(1, 20, n),
        "side": rng.choice(["buy", "sell"], n),
    })


# ---- 3.11 Liquidity cycles ----
def test_intraday_pattern_and_regime():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "minute_of_day": np.tile(np.arange(10), 20),
        "liquidity": rng.uniform(10, 100, 200),
    })
    eng = LiquidityCycleAnalyzer()
    pat = eng.analyze_intraday_liquidity_pattern(df)
    assert len(pat["best_execution_times"]) == 3
    regime = eng.predict_liquidity_regime({"current_liquidity": 20, "avg_liquidity": 100})
    assert regime["current_regime"] == "LIQUIDITY_CRISIS"


def test_liquidity_shocks():
    s = pd.Series([50] * 30 + [5] + [50] * 10)
    shocks = LiquidityCycleAnalyzer().detect_liquidity_shocks(s, threshold=2.0)
    assert len(shocks) >= 1


# ---- 3.12 Order flow predictor ----
def test_ofi_prediction_mean_reverts():
    rng = np.random.default_rng(1)
    # OFI mean-reverting alrededor de 0
    ofi = [0.0]
    for _ in range(200):
        ofi.append(0.5 * ofi[-1] + rng.normal(0, 1))
    preds = OrderFlowPredictor().predict_order_flow_imbalance(pd.Series(ofi), horizon=10)
    assert len(preds) == 10


def test_volume_forecast_and_arrival():
    ofp = OrderFlowPredictor()
    vol = pd.Series(np.tile([100, 200, 150, 300, 250], 10))
    fc = ofp.forecast_volume(vol, horizon=5)
    assert len(fc) == 5 and (fc > 0).all()
    arr = ofp.predict_order_arrival_rate(np.cumsum(np.random.default_rng(2).exponential(1, 50)))
    assert arr["predicted_arrival_rate"] >= 0


# ---- 3.13 Market resilience ----
def test_price_resilience_and_stability():
    rng = np.random.default_rng(3)
    prices = pd.Series(100 + np.cumsum(rng.normal(0, 0.05, 300)))
    eng = MarketResilienceAnalyzer()
    res = eng.measure_price_resilience(prices)
    assert 0 <= res["resilience_score"] <= 100
    stab = eng.assess_market_stability({"volatility": 0.1, "resilience_score": 80, "spread_stability": 0.9})
    assert stab["stability_regime"] in ("STABLE", "TRANSITIONAL", "UNSTABLE")


# ---- 3.14 Toxicity filter ----
def test_filter_toxic_flow_and_quote_adjust():
    tf = ToxicityFlowFilter()
    res = tf.filter_toxic_flow(_trades(), vpin_threshold=0.5)
    assert "toxic_fraction" in res
    prob = tf.detect_informed_trading_in_realtime(_trades())
    assert (prob.between(0, 1)).all()
    adj = tf.adjust_quotes_for_toxicity({"bid": 99.9, "ask": 100.1}, toxicity_score=1.0)
    assert (adj["ask"] - adj["bid"]) > 0.2  # spread ensanchado


# ---- 3.15 Market quality ----
def test_market_quality_metrics():
    trades = _trades()
    quotes = pd.DataFrame({"bid": trades["price"] - 0.02, "ask": trades["price"] + 0.02})
    mqm = MarketQualityMonitor()
    eff = mqm.calculate_effective_spread(trades, quotes)
    assert (eff >= 0).all()
    res = mqm.monitor_market_quality_metrics(trades, quotes)
    assert 0 <= res["quality_score"] <= 100
    assert "avg_effective_spread_bps" in res["quality_metrics"]
