import numpy as np
import pandas as pd

from quant_terminal.processing.microstructure import MarketMicrostructureAnalyzer
from quant_terminal.processing.order_book import OrderBookImbalanceAnalyzer
from quant_terminal.processing.pairs_trading import PairsTradingEngine
from quant_terminal.processing.stat_arb import StatisticalArbitrageEngine
from quant_terminal.processing.volatility_surface import VolatilitySurfaceAnalyzer


# ---- 1.1 Volatility surface ----
def test_iv_recovers_input_sigma():
    vs = VolatilitySurfaceAnalyzer(risk_free_rate=0.05)
    price = vs.black_scholes_price(100, 100, 0.5, 0.05, 0.25, "call")
    iv = vs.calculate_implied_volatility(price, 100, 100, 182.5, "call")
    assert abs(iv - 0.25) < 0.01


def test_build_surface_and_vrp():
    vs = VolatilitySurfaceAnalyzer()
    rows = []
    for K in (90, 100, 110):
        for dte in (30, 60):
            for typ in ("call", "put"):
                p = vs.black_scholes_price(100, K, dte / 365, 0.05, 0.2, typ)
                rows.append({"strike": K, "days_to_expiry": dte, "option_type": typ,
                             "bid": p * 0.99, "ask": p * 1.01, "underlying_price": 100})
    surface = vs.build_volatility_surface(pd.DataFrame(rows))
    assert 0.15 < surface["atm_iv"] < 0.25
    prices = list(100 + np.cumsum(np.random.default_rng(0).normal(0, 1, 100)))
    vrp = vs.calculate_volatility_risk_premium("X", pd.DataFrame(rows), prices)
    assert vrp["signal"] in ("SELL_VOL", "BUY_VOL", "NEUTRAL")


# ---- 1.2 Order book ----
def _book():
    return {
        "symbol": "AAPL",
        "bids": [(99.9, 500), (99.8, 300), (99.7, 200)],
        "asks": [(100.1, 100), (100.2, 80), (100.3, 60)],
    }


def test_imbalance_buy_signal():
    res = OrderBookImbalanceAnalyzer().calculate_bid_ask_imbalance(_book())
    assert res["imbalance"] > 0
    assert res["signal"] in ("BUY", "STRONG_BUY")
    assert res["mid_price"] == 100.0


def test_depth_score_grade():
    res = OrderBookImbalanceAnalyzer().calculate_market_depth_score(_book())
    assert 0 <= res["depth_score"] <= 100
    assert res["liquidity_grade"] in list("ABCDF")


# ---- 1.3 Microstructure ----
def test_vpin_range():
    rng = np.random.default_rng(1)
    trades = pd.DataFrame({
        "price": 100 + np.cumsum(rng.normal(0, 0.1, 500)),
        "volume": rng.integers(1, 10, 500),
        "side": rng.choice(["buy", "sell"], 500),
    })
    vpin = MarketMicrostructureAnalyzer().calculate_VPIN(trades, bucket_size=50)
    assert (vpin.dropna().between(0, 1)).all()


def test_price_impact_algo():
    res = MarketMicrostructureAnalyzer().estimate_price_impact(500000, daily_volume=1e7)
    assert res["total_impact_bps"] > 0
    assert res["recommended_algo"] in ("MARKET", "ICEBERG", "TWAP", "VWAP")


def test_optimal_spread_positive():
    res = MarketMicrostructureAnalyzer().calculate_optimal_spread(mid_price=100, sigma=0.02)
    assert res["optimal_spread_bps"] >= 0
    assert res["optimal_quote"]["ask"] >= res["optimal_quote"]["bid"]


# ---- 1.4 Pairs trading ----
def test_find_cointegrated_pairs():
    rng = np.random.default_rng(7)
    base = np.cumsum(rng.normal(0, 1, 300)) + 100
    a = base + rng.normal(0, 0.5, 300)
    b = base * 1.5 + rng.normal(0, 0.5, 300)
    c = np.cumsum(rng.normal(0, 1, 300)) + 100  # no cointegrado
    prices = pd.DataFrame({"A": a, "B": b, "C": c})
    pairs = PairsTradingEngine().find_cointegrated_pairs(prices)
    assert any(set(p["pair"]) == {"A", "B"} for p in pairs)


def test_monitor_pairs_signal():
    eng = PairsTradingEngine()
    pairs = [{"pair": ("A", "B"), "hedge_ratio": 1.0, "half_life": 10,
              "correlation": 0.9, "cointegration_pvalue": 0.01, "mu": 0.0, "sigma": 1.0}]
    sig = eng.monitor_pairs_in_realtime(pairs, {"A": 5.0, "B": 0.0})
    assert sig[0]["signal"] in ("ENTRY_SHORT", "STOP_LOSS")


# ---- 1.5 Stat arb ----
def test_cross_sectional_momentum():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2023-01-01", periods=300, freq="D")
    prices = pd.DataFrame({
        "WIN": 100 * np.exp(np.cumsum(rng.normal(0.002, 0.01, 300))),
        "LOSE": 100 * np.exp(np.cumsum(rng.normal(-0.002, 0.01, 300))),
        "FLAT": 100 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 300))),
    }, index=idx)
    mom = StatisticalArbitrageEngine().calculate_cross_sectional_momentum(
        prices, lookback=200, quantile=0.34
    )
    # El de mayor momentum es LONG; el de menor, SHORT.
    assert mom["momentum"].idxmax() == "WIN"
    assert mom.loc[mom["momentum"].idxmax(), "portfolio"] == "LONG"
    assert mom.loc[mom["momentum"].idxmin(), "portfolio"] == "SHORT"


def test_factor_model_and_portfolio():
    rng = np.random.default_rng(4)
    idx = pd.date_range("2023-01-01", periods=200, freq="D")
    factors = pd.DataFrame({"MKT": rng.normal(0, 0.01, 200), "SMB": rng.normal(0, 0.01, 200)}, index=idx)
    returns = pd.DataFrame({
        "X": 0.5 * factors["MKT"] + rng.normal(0, 0.005, 200),
        "Y": -0.3 * factors["SMB"] + rng.normal(0, 0.005, 200),
    }, index=idx)
    res = StatisticalArbitrageEngine().fit_factor_model(returns, factors)
    assert "factor_loadings" in res
    assert res["factor_loadings"].shape == (2, 2)
