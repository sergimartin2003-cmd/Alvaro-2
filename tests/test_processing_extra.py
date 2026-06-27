import pandas as pd

from quant_terminal.processing.carry_trade import CarryTradeAnalyzer
from quant_terminal.processing.fed_watch import FedWatchAnalyzer
from quant_terminal.processing.options_flow import OptionsFlowAnalyzer
from quant_terminal.processing.seasonality import SeasonalityAnalyzer
from quant_terminal.processing.vix_term_structure import VIXTermStructureAnalyzer


def test_put_call_ratio():
    data = [
        {"type": "C", "volume": 1000},
        {"type": "P", "volume": 1500},
    ]
    res = OptionsFlowAnalyzer().calculate_put_call_ratio(data)
    assert res["put_call_ratio"] == 1.5
    assert res["sentiment"] == "BULLISH"


def test_unusual_activity():
    chain = [{"symbol": "AAPL", "type": "C", "volume": 6000, "open_interest": 1000, "strike": 150}]
    sig = OptionsFlowAnalyzer().detect_unusual_activity(chain)
    assert sig and sig[0]["type"] == "UNUSUAL_VOLUME"


def test_vix_backwardation():
    a = VIXTermStructureAnalyzer()
    ts = a.calculate_term_structure({30: 35, 60: 30})
    assert ts["regime"] == "BACKWARDATION"
    action, _ = a.generate_signal(ts)
    assert action in ("BUY_EQUITIES", "BUY_VOLATILITY")


def test_fed_probabilities_sum():
    probs = FedWatchAnalyzer(current_rate=5.5).calculate_rate_probabilities({"2026-07": 94.5})
    p = probs["2026-07"]
    total = p["prob_cut"] + p["prob_hold"] + p["prob_hike"]
    assert abs(total - 1.0) < 1e-6


def test_carry_opportunities_sorted():
    opps = CarryTradeAnalyzer().calculate_carry_opportunities()
    assert opps
    assert opps[0]["annual_carry"] >= opps[-1]["annual_carry"]


def test_seasonality_monthly():
    idx = pd.date_range("2010-01-01", periods=2000, freq="D")
    import numpy as np

    prices = pd.Series(100 + np.cumsum(np.random.default_rng(1).normal(0, 1, 2000)), index=idx)
    sa = SeasonalityAnalyzer(pd.DataFrame({"SPY": prices}))
    monthly = sa.calculate_monthly_seasonality("SPY")
    assert len(monthly) == 12
