"""Demo del motor de ranking con datos sintéticos (offline, sin red).

Genera un pequeño universo de activos con distintas tendencias, ejecuta el
RealTimeRankingEngine y muestra el ranking, el resumen de mercado y un preview
del mensaje de alerta de Telegram para la mejor oportunidad.

Ejecutar: PYTHONPATH=. python examples/run_ranking_demo.py
"""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd

from quant_terminal.alerts.telegram_alerts import TelegramAlertSystem
from quant_terminal.analysis.real_time_ranking import RealTimeRankingEngine


def synth(seed: int, trend: float, n: int = 220) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(trend, 0.012, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    vol = rng.integers(1_000_000, 5_000_000, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


def main() -> None:
    config = {
        "assets_universe": {
            "stocks": {"mega": ["AAPL", "MSFT", "TSLA", "NVDA"]},
            "crypto": {"top": ["BTC/USD", "ETH/USD"]},
        }
    }
    trends = {
        "AAPL": (1, 0.0030), "MSFT": (2, 0.0008), "TSLA": (3, -0.0030),
        "NVDA": (5, 0.0040), "BTC/USD": (4, 0.0015), "ETH/USD": (6, -0.0010),
    }

    def provider(symbol: str, asset_class: str) -> pd.DataFrame:
        seed, trend = trends[symbol]
        return synth(seed, trend)

    # Contexto sintético opcional (sentimiento/opciones) para enriquecer scores.
    def context(symbol: str, asset_class: str) -> dict:
        base = {"AAPL": 80, "NVDA": 85, "BTC/USD": 70}.get(symbol, 50)
        return {"sentiment_score": base, "options_flow_score": base, "ml_prediction_score": base}

    engine = RealTimeRankingEngine(config, data_provider=provider, context_provider=context)
    asyncio.run(engine.analyze_all_assets())

    summary = engine.get_market_summary()
    print("=== RESUMEN DE MERCADO ===")
    print(f"  Activos: {summary['total_assets_analyzed']}  Sentimiento: {summary['market_sentiment']}"
          f"  Score medio: {summary['average_score']:.1f}")
    print(f"  Alcistas/Bajistas/Neutrales: {summary['bullish_assets']}/"
          f"{summary['bearish_assets']}/{summary['neutral_assets']}\n")

    print("=== RANKING COMPLETO ===")
    ranked = sorted(engine.current_rankings.values(), key=lambda a: a.final_score, reverse=True)
    for a in ranked:
        print(f"  {a.symbol:8s} {a.final_score:5.1f}  {a.action:11s} "
              f"(téc {a.technical_score:.0f} / sent {a.sentiment_score:.0f})")

    best = ranked[0]
    print(f"\n=== PREVIEW ALERTA TELEGRAM ({best.symbol}) ===")
    print(TelegramAlertSystem.format_strong_buy_alert(best))


if __name__ == "__main__":
    main()
