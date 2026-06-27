"""Demo offline de Jarvis: briefing diario + conversación, con datos sintéticos.

No requiere red, claves ni LLM: el advisor usa un resumen de plantilla y el
asistente responde con sus handlers. Si defines ANTHROPIC_API_KEY e instalas el
SDK `anthropic`, el resumen y las preguntas libres pasarán a usar Claude.

Ejecutar: PYTHONPATH=. python examples/run_jarvis_demo.py
"""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd

from quant_terminal.analysis.real_time_ranking import RealTimeRankingEngine
from quant_terminal.jarvis.assistant import JarvisAssistant
from quant_terminal.jarvis.daily_advisor import DailyMarketAdvisor
from quant_terminal.jarvis.llm import LLMClient


def synth(seed: int, trend: float, n: int = 220) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(trend, 0.012, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    vol = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol})


async def main() -> None:
    config = {"assets_universe": {"stocks": {"mega": ["AAPL", "NVDA", "TSLA", "MSFT"]}}}
    trends = {"AAPL": (1, 0.004), "NVDA": (5, 0.005), "TSLA": (3, -0.004), "MSFT": (2, 0.0008)}

    def ctx(symbol, asset_class):
        boost = {"AAPL": 88, "NVDA": 90}.get(symbol, 55)
        return {"sentiment_score": boost, "options_flow_score": boost, "ml_prediction_score": boost}

    engine = RealTimeRankingEngine(config, data_provider=lambda s, c: synth(*trends[s]),
                                   context_provider=ctx)
    await engine.analyze_all_assets()

    llm = LLMClient()
    advisor = DailyMarketAdvisor({"user_name": "Álvaro"}, ranking_engine=engine, llm_client=llm)
    assistant = JarvisAssistant({"user_name": "Álvaro"}, daily_advisor=advisor,
                                llm_client=llm, ranking_engine=engine)

    print(f"(LLM Claude disponible: {llm.available})\n")
    print("=== BRIEFING DIARIO ===")
    briefing = await advisor.generate_daily_briefing()
    print(advisor.format_telegram_briefing(briefing).replace("<b>", "").replace("</b>", ""))

    print("\n=== CONVERSACIÓN ===")
    for q in ["sentimiento del mercado", "dame el top 3 activos", "analiza AAPL", "qué debo hacer"]:
        print(f"\nTú: {q}")
        print("Jarvis:", await assistant.process_text_input(q))


if __name__ == "__main__":
    asyncio.run(main())
