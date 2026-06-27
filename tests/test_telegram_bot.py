import asyncio

import numpy as np
import pandas as pd

from quant_terminal.analysis.real_time_ranking import RealTimeRankingEngine
from quant_terminal.jarvis.assistant import JarvisAssistant
from quant_terminal.jarvis.daily_advisor import DailyMarketAdvisor
from quant_terminal.jarvis.telegram_bot import JarvisTelegramBot


def _synth(seed, trend, n=220):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(trend, 0.012, n)))
    h = c * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = c * (1 - np.abs(rng.normal(0, 0.003, n)))
    o = np.concatenate([[c[0]], c[:-1]])
    v = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame({"open": o, "high": h, "low": low, "close": c, "volume": v})


def _bot():
    cfg = {"assets_universe": {"stocks": {"m": ["AAPL", "TSLA", "MSFT"]}}}
    seeds = {"AAPL": (1, 0.004), "TSLA": (3, -0.004), "MSFT": (2, 0.0005)}
    ctx = lambda s, c: {"sentiment_score": 88 if s == "AAPL" else 55,
                        "options_flow_score": 88 if s == "AAPL" else 55,
                        "ml_prediction_score": 85 if s == "AAPL" else 55}
    eng = RealTimeRankingEngine(cfg, data_provider=lambda s, c: _synth(*seeds[s]), context_provider=ctx)
    asyncio.run(eng.analyze_all_assets())
    advisor = DailyMarketAdvisor(ranking_engine=eng)
    assistant = JarvisAssistant(daily_advisor=advisor, ranking_engine=eng)
    return JarvisTelegramBot({"telegram": {"bot_token": "TESTTOKEN"}}, assistant=assistant,
                             daily_advisor=advisor, ranking_engine=eng)


# ---- parsing ----
def test_parse_command():
    p = JarvisTelegramBot.parse_command
    assert p("/ranking 5") == ("ranking", "5")
    assert p("/analyze AAPL") == ("analyze", "AAPL")
    assert p("/help") == ("help", "")
    assert p("/analyze@MiBot TSLA") == ("analyze", "TSLA")
    assert p("hola jarvis") == (None, "hola jarvis")


def test_parse_int():
    assert JarvisTelegramBot._parse_int("3", 5) == 3
    assert JarvisTelegramBot._parse_int("", 5) == 5
    assert JarvisTelegramBot._parse_int("abc", 5) == 5


# ---- dispatch (offline) ----
def test_handle_commands():
    bot = _bot()
    assert "Comandos de Jarvis" in asyncio.run(bot.handle("/help"))
    assert "Comandos de Jarvis" in asyncio.run(bot.handle("/start"))

    ranking = asyncio.run(bot.handle("/ranking 2"))
    assert "OPORTUNIDADES" in ranking.upper()

    analysis = asyncio.run(bot.handle("/analyze AAPL"))
    assert "AAPL" in analysis

    sentiment = asyncio.run(bot.handle("/sentiment"))
    assert "Sentimiento" in sentiment

    briefing = asyncio.run(bot.handle("/briefing"))
    assert "BRIEFING DIARIO" in briefing


def test_handle_unknown_and_freetext():
    bot = _bot()
    assert "desconocido" in asyncio.run(bot.handle("/foobar")).lower()
    # texto libre se enruta al asistente (intención -> sentimiento)
    free = asyncio.run(bot.handle("sentimiento del mercado"))
    assert "Sentimiento" in free


def test_analyze_without_symbol():
    bot = _bot()
    assert "Uso:" in asyncio.run(bot.handle("/analyze"))
