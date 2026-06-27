import asyncio

import numpy as np
import pandas as pd

from quant_terminal.analysis.real_time_ranking import RealTimeRankingEngine


def _synth(seed, trend=0.0008):
    rng = np.random.default_rng(seed)
    n = 220
    rets = rng.normal(trend, 0.012, n)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    vol = rng.integers(1_000_000, 5_000_000, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


def _engine():
    config = {
        "assets_universe": {
            "stocks": {"mega": ["AAPL", "MSFT", "TSLA"]},
            "crypto": {"top": ["BTC/USD"]},
        }
    }
    seeds = {"AAPL": (1, 0.002), "MSFT": (2, 0.0008), "TSLA": (3, -0.002), "BTC/USD": (4, 0.001)}

    def provider(symbol, asset_class):
        seed, trend = seeds[symbol]
        return _synth(seed, trend)

    return RealTimeRankingEngine(config, data_provider=provider)


def test_universe_flattened():
    eng = _engine()
    assert ("AAPL", "stocks") in eng.universe
    assert ("BTC/USD", "crypto") in eng.universe
    assert len(eng.universe) == 4


def test_analyze_single_asset():
    eng = _engine()
    a = asyncio.run(eng.analyze_single_asset("AAPL", "stocks"))
    assert 0 <= a.final_score <= 100
    assert a.action in ("STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL")
    assert isinstance(a.reasons_to_buy, list)
    assert a.entry_price > 0 and a.stop_loss > 0


def test_analyze_all_and_summary():
    eng = _engine()
    asyncio.run(eng.analyze_all_assets())
    assert len(eng.current_rankings) == 4
    summary = eng.get_market_summary()
    assert summary["total_assets_analyzed"] == 4
    assert summary["market_sentiment"] in ("BULLISH", "BEARISH", "NEUTRAL")
    assert "by_asset_class" in summary
    # El ranking ordena por score descendente.
    top = eng.get_top_opportunities(4, min_score=0)
    scores = [a.final_score for a in top]
    assert scores == sorted(scores, reverse=True)


def test_weights_normalized():
    eng = _engine()
    assert abs(sum(eng.weights.values()) - 1.0) < 1e-9


def test_significant_change_detection():
    eng = _engine()
    asyncio.run(eng.analyze_all_assets())
    prev = dict(eng.current_rankings)
    # Forzar un cambio grande de score en un activo.
    aapl = prev["AAPL"]
    bumped = {**eng.current_rankings}
    import copy

    new_aapl = copy.copy(aapl)
    new_aapl.final_score = aapl.final_score + 25
    new_aapl.action = "STRONG_BUY"
    bumped["AAPL"] = new_aapl
    events = eng.detect_significant_changes(prev, bumped)
    assert any(e["symbol"] == "AAPL" for e in events)
