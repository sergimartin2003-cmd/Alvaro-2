from quant_terminal.aggregation.bayesian import BayesianDecisionEngine
from quant_terminal.aggregation.signal_aggregator import SignalAggregator


def _signals(direction="BUY"):
    return {
        "technical_analysis": {"signal": direction, "confidence": 0.8, "strength": 0.8},
        "sentiment_analysis": {"signal": direction, "confidence": 0.7, "strength": 0.7},
        "options_flow": {"signal": direction, "confidence": 0.75, "strength": 0.7},
    }


def test_aggregate_buy():
    res = SignalAggregator().aggregate_signals(_signals("BUY"))
    assert res["aggregate_signal"] == "BUY"
    assert res["aggregate_score"] > 0


def test_conflict_detection():
    sig = _signals("BUY")
    sig["macro_analysis"] = {"signal": "SELL", "confidence": 0.9, "strength": 0.9}
    res = SignalAggregator().aggregate_signals(sig)
    assert res["confluence_analysis"]["conflict_detected"] is True


def test_weight_update_normalizes():
    agg = SignalAggregator()
    agg.update_source_weights(
        {"technical_analysis": {"wins": 10, "losses": 2, "total_pnl": 0.2}}
    )
    assert all(0 <= s["weight"] <= 1 for s in agg.signal_sources.values())


def test_bayesian_update_buy():
    res = BayesianDecisionEngine().bayesian_update(_signals("BUY"))
    assert res["posterior_probabilities"]["market_up"] > 0.5
    assert res["decision"]["action"] in ("BUY", "STRONG_BUY")
