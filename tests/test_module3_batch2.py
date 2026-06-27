import numpy as np
import pandas as pd

from quant_terminal.microstructure.flash_crash import FlashCrashDetector
from quant_terminal.microstructure.latency_arbitrage import LatencyArbitrageDetector
from quant_terminal.microstructure.optimal_execution import OptimalExecutionEngine
from quant_terminal.microstructure.price_discovery import PriceDiscoveryModel
from quant_terminal.microstructure.smart_order_router import SmartOrderRouter


# ---- 3.6 Price discovery ----
def test_information_share_and_efficiency():
    rng = np.random.default_rng(0)
    p1 = 100 + np.cumsum(rng.normal(0, 1, 200))      # más volátil => más share
    p2 = 100 + np.cumsum(rng.normal(0, 0.2, 200))
    pdm = PriceDiscoveryModel()
    res = pdm.calculate_information_share({"NYSE": p1, "BATS": p2})
    assert res["dominant_market"] == "NYSE"
    eff = pdm.measure_market_efficiency(pd.Series(p1))
    assert "variance_ratio" in eff
    assert eff["predictability"] in ("MEAN_REVERTING", "TRENDING", "EFFICIENT")


def test_lead_lag():
    rng = np.random.default_rng(1)
    base = np.cumsum(rng.normal(0, 1, 300))
    a1 = pd.Series(100 + base)
    a2 = pd.Series(100 + np.r_[0, 0, base[:-2]])  # a2 sigue a a1 con lag 2
    res = PriceDiscoveryModel().detect_lead_lag_relationships(a1, a2, max_lag=5)
    assert res["optimal_lag"] >= 1


# ---- 3.7 Optimal execution ----
def test_twap_vwap_split():
    eng = OptimalExecutionEngine()
    twap = eng.execute_twap(1000, 4)
    assert abs(sum(twap["execution_schedule"]) - 1000) < 1e-6
    vwap = eng.execute_vwap(1000, pd.Series([1, 2, 3, 4]))
    assert abs(sum(vwap["execution_schedule"]) - 1000) < 1e-6
    assert vwap["execution_schedule"][-1] > vwap["execution_schedule"][0]  # más volumen al final


def test_is_and_pov():
    eng = OptimalExecutionEngine()
    is_res = eng.execute_implementation_shortfall(1000, horizon=10, risk_aversion=2.0)
    assert abs(sum(is_res["execution_schedule"]) - 1000) < 1e-3
    pov = eng.execute_pov(1000, pd.Series([5000] * 10), participation_rate=0.1)
    assert sum(pov["execution_schedule"]) <= 1000 + 1e-6


# ---- 3.8 Smart order router ----
def test_select_and_split_venue():
    sor = SmartOrderRouter()
    venues = {
        "A": {"ask": 100.05, "bid": 99.95, "liquidity": 3000, "fee_bps": 3, "latency_ms": 5},
        "B": {"ask": 100.02, "bid": 99.98, "liquidity": 8000, "fee_bps": 1, "latency_ms": 1},
    }
    best = sor.select_best_venue({"side": "BUY"}, venues)
    assert best["best_venue"] == "B"  # mejor en precio, liquidez, fees y latencia
    split = sor.split_order_across_venues(1000, venues)
    assert abs(sum(split["venue_allocations"].values()) - 1000) < 1e-6


def test_monitor_execution_quality():
    sor = SmartOrderRouter()
    fills = [{"price": 100.1, "venue": "A"}, {"price": 100.2, "venue": "B"}]
    res = sor.monitor_execution_quality(fills, benchmark_price=100.0)
    assert res["avg_slippage_bps"] > 0
    assert "A" in res["venue_performance"]


# ---- 3.9 Latency arbitrage ----
def test_cross_venue_and_advantage():
    det = LatencyArbitrageDetector()
    quotes = {
        "V1": {"bid": 100.10, "ask": 100.15, "latency_ms": 2},
        "V2": {"bid": 100.00, "ask": 100.05, "latency_ms": 1},
    }
    arbs = det.detect_cross_venue_arbitrage(quotes)
    assert arbs and arbs[0]["buy_venue"] == "V2" and arbs[0]["sell_venue"] == "V1"
    adv = det.measure_latency_advantage(1.0, 10.0)
    assert adv["advantage_level"] in ("SIGNIFICANT", "DOMINANT")
    assert adv["exploitable"] is True


def test_stale_quotes():
    quotes = pd.DataFrame({"age_ms": [50, 200], "expected_move": [0.0, 0.05]})
    res = LatencyArbitrageDetector().detect_stale_quotes(quotes, max_latency_ms=100)
    assert len(res) == 1


# ---- 3.10 Flash crash ----
def test_detect_flash_crash():
    prices = pd.Series([100] * 10 + [100, 97, 94, 92, 90] + [95, 99, 100] * 3)
    res = FlashCrashDetector().detect_flash_crash(prices, window=5, threshold_pct=-5)
    assert res["is_flash_crash"] is True
    assert res["crash_magnitude"] < -5


def test_predict_flash_crash_risk():
    res = FlashCrashDetector().predict_flash_crash_risk(
        {"liquidity_score": 20, "volatility": 0.5, "order_imbalance": 0.8}
    )
    assert res["crash_risk_score"] > 0.5
    assert res["recommended_actions"]
