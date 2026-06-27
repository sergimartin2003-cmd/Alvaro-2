import asyncio

import numpy as np
import pandas as pd

from quant_terminal.analysis.real_time_ranking import RealTimeRankingEngine
from quant_terminal.jarvis.assistant import JarvisAssistant
from quant_terminal.jarvis.daily_advisor import DailyMarketAdvisor
from quant_terminal.jarvis.llm import LLMClient


def _synth(seed, trend):
    rng = np.random.default_rng(seed)
    n = 220
    c = 100 * np.exp(np.cumsum(rng.normal(trend, 0.012, n)))
    h = c * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = c * (1 - np.abs(rng.normal(0, 0.003, n)))
    o = np.concatenate([[c[0]], c[:-1]])
    v = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame({"open": o, "high": h, "low": low, "close": c, "volume": v})


def _engine():
    cfg = {"assets_universe": {"stocks": {"m": ["AAPL", "TSLA", "MSFT"]}}}
    seeds = {"AAPL": (1, 0.004), "TSLA": (3, -0.004), "MSFT": (2, 0.0005)}

    def ctx(symbol, asset_class):
        return {"sentiment_score": {"AAPL": 88}.get(symbol, 55),
                "options_flow_score": {"AAPL": 90}.get(symbol, 55),
                "ml_prediction_score": {"AAPL": 85}.get(symbol, 55)}

    eng = RealTimeRankingEngine(cfg, data_provider=lambda s, c: _synth(*seeds[s]),
                                context_provider=ctx)
    asyncio.run(eng.analyze_all_assets())
    return eng


# ---- LLM client (offline) ----
def test_llm_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = LLMClient(api_key=None)
    assert client.available is False


# ---- Intent detection ----
def test_intent_detection():
    a = JarvisAssistant()
    assert a.detect_intent("dame el briefing diario")["command"] == "daily_briefing"
    assert a.detect_intent("analiza AAPL")["command"] == "analyze_asset"
    assert a.detect_intent("analiza AAPL")["parameters"]["symbol"] == "AAPL"
    top = a.detect_intent("dame el top 5 activos")
    assert top["command"] == "top_assets"
    assert top["parameters"]["n"] == 5
    assert a.detect_intent("qué riesgos hay")["command"] == "risk_warnings"
    assert a.detect_intent("hola, cómo estás")["command"] == "general"


# ---- Daily advisor (offline template summary) ----
def test_daily_briefing_offline():
    eng = _engine()
    advisor = DailyMarketAdvisor({"user_name": "Álvaro"}, ranking_engine=eng, llm_client=None)
    briefing = asyncio.run(advisor.generate_daily_briefing())
    assert "Álvaro" in briefing.market_summary
    assert isinstance(briefing.top_buys, list)
    assert isinstance(briefing.action_items, list)
    # Formato Telegram contiene cabecera
    tg = advisor.format_telegram_briefing(briefing)
    assert "BRIEFING DIARIO" in tg


def test_top_buys_mapping():
    eng = _engine()
    advisor = DailyMarketAdvisor(ranking_engine=eng)
    buys = asyncio.run(advisor._get_top_buys(10))
    # Solo activos con score >= 70 (puede ser 0 según datos), estructura correcta
    for b in buys:
        assert {"symbol", "score", "entry_price", "stop_loss", "take_profit"} <= set(b)
        assert b["score"] >= 70


# ---- Assistant handlers (async, offline) ----
def test_assistant_handlers():
    eng = _engine()
    advisor = DailyMarketAdvisor(ranking_engine=eng)
    assistant = JarvisAssistant(daily_advisor=advisor, ranking_engine=eng, llm_client=None)

    sentiment = asyncio.run(assistant.process_text_input("sentimiento del mercado"))
    assert "Sentimiento" in sentiment

    analysis = asyncio.run(assistant.process_text_input("analiza AAPL"))
    assert "AAPL" in analysis

    help_txt = asyncio.run(assistant.process_text_input("ayuda"))
    assert "briefing" in help_txt.lower()

    # Query general sin LLM => mensaje de fallback
    general = asyncio.run(assistant.process_text_input("cuéntame un chiste"))
    assert "LLM" in general or "ayuda" in general.lower()


def test_closing_summary():
    eng = _engine()
    advisor = DailyMarketAdvisor(ranking_engine=eng)
    text = advisor.format_closing_summary()
    assert "CIERRE DE MERCADO" in text
    assert "Sentimiento" in text


def test_conversation_history_tracked():
    assistant = JarvisAssistant()
    asyncio.run(assistant.process_text_input("ayuda"))
    assert len(assistant.conversation_history) == 2
    assert assistant.conversation_history[0]["role"] == "user"
