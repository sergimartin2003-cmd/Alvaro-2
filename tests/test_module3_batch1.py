import numpy as np
import pandas as pd

from quant_terminal.microstructure.liquidity_provider import LiquidityProvider
from quant_terminal.microstructure.market_impact import MarketImpactModel
from quant_terminal.microstructure.order_book_dynamics import OrderBookDynamics
from quant_terminal.microstructure.order_flow_toxicity import OrderFlowToxicityAnalyzer
from quant_terminal.microstructure.trade_classification import TradeClassification


def _trades(seed=0, n=400):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "price": 100 + np.cumsum(rng.normal(0, 0.05, n)),
        "volume": rng.integers(1, 20, n),
        "side": rng.choice(["buy", "sell"], n),
    })


# ---- 3.1 Toxicity ----
def test_vpin_and_traders():
    eng = OrderFlowToxicityAnalyzer()
    vpin = eng.calculate_VPIN(_trades(), bucket_size=50)
    assert (vpin.dropna().between(0, 1)).all()
    cls = eng.classify_traders(_trades())
    assert 0 <= cls["pin"] <= 1
    assert 0 <= cls["informed_flow_percentage"] <= 1


def test_adverse_selection():
    trades = _trades()
    quotes = pd.DataFrame({"bid": trades["price"] - 0.01, "ask": trades["price"] + 0.01})
    res = OrderFlowToxicityAnalyzer().detect_adverse_selection(quotes, trades)
    assert 0 <= res["adverse_selection_score"] <= 1
    assert res["recommendation"] in ("WIDEN_SPREAD", "REDUCE_SIZE", "NORMAL")


# ---- 3.2 Market impact ----
def test_optimal_execution_decreasing():
    res = MarketImpactModel().calculate_optimal_execution(10000, horizon=10, risk_aversion=1.0)
    remaining = res["remaining_trajectory"]
    assert remaining[0] >= remaining[-1]  # cantidad pendiente decrece
    assert abs(remaining[-1]) < 1e-6
    assert res["expected_cost"] >= 0


def test_simulate_impact_scales_with_size():
    m = MarketImpactModel()
    small = m.simulate_market_impact(1e5)["total_impact_bps"]
    big = m.simulate_market_impact(1e6)["total_impact_bps"]
    assert big > small


# ---- 3.3 Liquidity provider ----
def test_optimal_quotes_inventory_skew():
    lp = LiquidityProvider()
    flat = lp.calculate_optimal_quotes(100, inventory=0)
    long = lp.calculate_optimal_quotes(100, inventory=50)
    # Con inventario largo, el reservation price baja (incentiva vender).
    assert long["reservation_price"] < flat["reservation_price"]
    assert flat["ask_price"] > flat["bid_price"]


def test_simulate_market_making():
    prices = pd.Series(100 + np.cumsum(np.random.default_rng(1).normal(0, 0.1, 200)))
    res = LiquidityProvider().simulate_market_making_strategy(prices, seed=0)
    assert res["n_trades"] > 0
    assert "sharpe_ratio" in res
    assert len(res["equity_curve"]) == len(prices)


# ---- 3.4 Order book dynamics ----
def test_queue_dynamics_and_cancel_patterns():
    rng = np.random.default_rng(2)
    n = 300
    updates = pd.DataFrame({
        "action": rng.choice(["add", "cancel", "execute"], n, p=[0.5, 0.4, 0.1]),
        "lifetime_ms": rng.integers(10, 2000, n),
        "volume": rng.integers(1, 100, n),
    })
    eng = OrderBookDynamics()
    dyn = eng.model_queue_dynamics(updates)
    assert 0 <= dyn["execution_probability"] <= 1
    pat = eng.detect_cancelation_patterns(updates)
    assert 0 <= pat["spoofing_probability"] <= 1


def test_predict_book_evolution():
    book = {"bids": [(99.9, 100)], "asks": [(100.1, 100)]}
    res = OrderBookDynamics().predict_order_book_evolution(book, imbalance=0.8)
    assert res["expected_mid_price_change"] > 0  # imbalance alcista sube el mid


# ---- 3.5 Trade classification ----
def test_lee_ready_and_tick_rule():
    trades = pd.DataFrame({"price": [100.0, 100.5, 100.2, 100.2], "volume": [10, 20, 5, 8]})
    quotes = pd.DataFrame({"bid": [99.9, 100.0, 100.0, 100.0], "ask": [100.1, 100.4, 100.4, 100.4]})
    tc = TradeClassification()
    lr = tc.classify_trades_lee_ready(trades, quotes)
    assert lr.iloc[1] == "BUY"  # 100.5 > mid
    tick = tc.classify_trades_tick_rule(trades)
    assert tick.iloc[1] == "BUY"  # uptick
    assert tick.iloc[2] == "SELL"  # downtick


def test_ofi():
    classified = pd.DataFrame({"side": ["BUY", "BUY", "SELL"], "volume": [10, 20, 5]})
    ofi = TradeClassification().calculate_order_flow_imbalance(classified)
    assert ofi.iloc[-1] == 25  # 10 + 20 - 5
