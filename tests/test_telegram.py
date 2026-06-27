import asyncio
from datetime import datetime

from quant_terminal.alerts.telegram_alerts import TelegramAlertSystem
from quant_terminal.analysis.real_time_ranking import AssetAnalysis


def _analysis(score=87.5):
    return AssetAnalysis(
        symbol="AAPL",
        asset_class="stocks",
        current_price=150.25,
        timestamp=datetime(2026, 6, 27, 14, 30, 0),
        final_score=score,
        action="STRONG_BUY",
        confidence=0.85,
        expected_return_24h=3.5,
        expected_return_7d=8.2,
        risk_reward_ratio=2.8,
        entry_price=150.25,
        stop_loss=148.50,
        take_profit_1=153.5,
        take_profit_2=156.0,
        take_profit_3=160.0,
        technical_score=85,
        sentiment_score=88,
        reasons_to_buy=["✅ Tendencia alcista fuerte", "✅ MACD alcista"],
        reasons_to_avoid=["❌ Resistencia cercana"],
        key_catalysts=["📅 FOMC - 2026-07-31 (Impacto: High)"],
        risk_factors=["⚠️ Euforia posible"],
    )


def test_format_strong_buy_contains_key_fields():
    msg = TelegramAlertSystem.format_strong_buy_alert(_analysis())
    assert "AAPL" in msg
    assert "87.5/100" in msg
    assert "Tendencia alcista" in msg
    assert "Stop Loss: $148.50" in msg
    assert "FOMC" in msg


def test_format_daily_summary():
    summary = {
        "timestamp": datetime(2026, 6, 27, 9, 0, 0),
        "total_assets_analyzed": 20,
        "market_sentiment": "BULLISH",
        "bullish_assets": 12,
        "bearish_assets": 5,
        "neutral_assets": 3,
        "top_opportunities": [{"symbol": "AAPL", "score": 87.5, "signal": "STRONG_BUY"}],
        "top_risks": [{"symbol": "META", "score": 18.5, "signal": "STRONG_SELL"}],
        "by_asset_class": {"stocks": {"avg_score": 72.5, "bullish_pct": 0.62, "top": "AAPL"}},
    }
    msg = TelegramAlertSystem.format_daily_summary(summary)
    assert "RESUMEN DIARIO" in msg
    assert "BULLISH" in msg
    assert "AAPL" in msg and "META" in msg


def test_rate_limiting_blocks_duplicates():
    tg = TelegramAlertSystem({"chat_id": None, "min_interval_seconds": 0, "dedup_window_seconds": 3600})
    # Sin chat_ids, _post no se llama; el control de rate limiting sí.
    first = asyncio.run(tg.send("hola", symbol="AAPL", priority="HIGH"))
    second = asyncio.run(tg.send("otra", symbol="AAPL", priority="HIGH"))
    assert first["sent"] is True
    assert second["sent"] is False  # deduplicado por símbolo


def test_extreme_bypasses_dedup():
    tg = TelegramAlertSystem({"chat_id": None})
    asyncio.run(tg.send("a", symbol="AAPL", priority="HIGH"))
    extreme = asyncio.run(tg.send("b", symbol="AAPL", priority="EXTREME"))
    assert extreme["sent"] is True
