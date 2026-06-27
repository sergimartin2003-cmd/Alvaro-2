"""El modo demo de la terminal debe poblarse sin claves ni red."""

import asyncio

import main


def test_demo_orchestrator_populates_rankings():
    orch = main.TradingSystemOrchestrator(config_path="config/__missing__.yaml", demo=True)
    assert len(orch.ranking_engine.universe) > 0
    asyncio.run(orch.ranking_engine.analyze_all_assets())
    summary = orch.ranking_engine.get_market_summary()
    assert summary["total_assets_analyzed"] > 0
    assert summary["top_opportunities"]  # tablas no vacías
    # Hay variedad de señales (no todo NEUTRAL)
    actions = {a.action for a in orch.ranking_engine.current_rankings.values()}
    assert actions - {"HOLD"}


def test_synthetic_ohlcv_shape():
    df = main._synthetic_ohlcv("AAPL", n=120)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 120
    assert (df["high"] >= df["low"]).all()


def test_demo_default_config_has_universe():
    cfg = main._demo_default_config()
    assert "assets_universe" in cfg
    assert "stocks" in cfg["assets_universe"]
