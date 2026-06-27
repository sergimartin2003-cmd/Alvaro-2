"""Motor de ranking en tiempo real.

Analiza un universo de activos, combina las salidas de las capas existentes
(processing/aggregation) en un score 0-100 por categoría, produce un score final
ponderado, una acción (STRONG_BUY..STRONG_SELL) y explicaciones accionables
(razones para comprar/evitar, catalizadores, factores de riesgo).

Diseño para funcionar offline: los datos OHLCV se obtienen mediante un
``data_provider`` inyectable y el contexto no técnico (sentimiento, macro,
opciones, ML, estacionalidad) mediante un ``context_provider`` opcional. Si no
hay contexto para una categoría, se usa un valor neutro (50) para no sesgar.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from ..processing.signal_generator import SignalGenerator
from ..processing.technical_indicators import TechnicalIndicatorEngine

logger = logging.getLogger(__name__)

# Pesos por defecto (deben sumar 1.0).
_DEFAULT_WEIGHTS = {
    "technical": 0.25,
    "sentiment": 0.20,
    "macro": 0.15,
    "options_flow": 0.15,
    "ml_prediction": 0.15,
    "seasonality": 0.05,
    "volume_profile": 0.03,
    "correlation": 0.02,
}

_DEFAULT_THRESHOLDS = {"strong_buy": 75, "buy": 60, "sell": 40, "strong_sell": 25}

_NEUTRAL = 50.0


@dataclass
class AssetAnalysis:
    """Resultado completo del análisis de un activo."""

    symbol: str
    asset_class: str
    current_price: float
    timestamp: datetime
    technical_score: float = _NEUTRAL
    sentiment_score: float = _NEUTRAL
    macro_score: float = _NEUTRAL
    options_flow_score: float = _NEUTRAL
    ml_prediction_score: float = _NEUTRAL
    seasonality_score: float = _NEUTRAL
    volume_profile_score: float = _NEUTRAL
    correlation_score: float = _NEUTRAL
    final_score: float = _NEUTRAL
    signal_strength: str = "NEUTRAL"
    action: str = "HOLD"
    confidence: float = 0.0
    expected_return_24h: float = 0.0
    expected_return_7d: float = 0.0
    risk_reward_ratio: float = 0.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit_1: float = 0.0
    take_profit_2: float = 0.0
    take_profit_3: float = 0.0
    reasons_to_buy: list[str] = field(default_factory=list)
    reasons_to_avoid: list[str] = field(default_factory=list)
    key_catalysts: list[str] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)
    technical_indicators: dict = field(default_factory=dict)
    sentiment_data: dict = field(default_factory=dict)
    macro_events: list = field(default_factory=list)
    options_activity: dict = field(default_factory=dict)
    recent_news: list = field(default_factory=list)
    social_mentions: int = 0
    volume_anomaly: float = 1.0

    def to_row(self) -> dict:
        """Fila compacta para tablas/dashboards."""
        return {
            "symbol": self.symbol,
            "asset_class": self.asset_class,
            "score": round(self.final_score, 1),
            "signal": self.action,
            "price": round(self.current_price, 2),
            "expected_24h": round(self.expected_return_24h, 2),
            "risk_reward": round(self.risk_reward_ratio, 2),
            "entry": round(self.entry_price, 2),
            "stop_loss": round(self.stop_loss, 2),
            "take_profit_2": round(self.take_profit_2, 2),
            "confidence": round(self.confidence, 2),
        }


class RealTimeRankingEngine:
    """Corazón del sistema: rankea todos los activos del universo."""

    def __init__(self, config: dict | None = None, data_provider=None,
                 context_provider=None) -> None:
        config = config or {}
        rcfg = config.get("ranking_engine", {})
        self.weights = {**_DEFAULT_WEIGHTS, **rcfg.get("weights", {})}
        self._normalize_weights()
        self.thresholds = {**_DEFAULT_THRESHOLDS, **rcfg.get("alert_thresholds", {})}
        self.universe = self._flatten_universe(config.get("assets_universe", {}))

        self.technical_engine = TechnicalIndicatorEngine()
        self.signal_generator = SignalGenerator(self.technical_engine)

        # Proveedores inyectables (permiten correr offline / en tests).
        self.data_provider = data_provider
        self.context_provider = context_provider

        self.current_rankings: dict[str, AssetAnalysis] = {}
        self.historical_rankings: list[dict] = []

    # ------------------------------------------------------------------ setup
    def _normalize_weights(self) -> None:
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    @staticmethod
    def _flatten_universe(universe: dict) -> list[tuple[str, str]]:
        """Convierte el árbol de config en pares (symbol, asset_class)."""
        out: list[tuple[str, str]] = []
        for asset_class, groups in universe.items():
            if isinstance(groups, dict):
                for symbols in groups.values():
                    out.extend((s, asset_class) for s in symbols)
            elif isinstance(groups, list):
                out.extend((s, asset_class) for s in groups)
        return out

    # --------------------------------------------------------------- análisis
    async def analyze_all_assets(self) -> dict[str, AssetAnalysis]:
        """Analiza todo el universo en paralelo (un fallo no detiene el resto)."""
        previous = dict(self.current_rankings)
        results = await asyncio.gather(
            *(self.analyze_single_asset(sym, cls) for sym, cls in self.universe),
            return_exceptions=True,
        )
        rankings: dict[str, AssetAnalysis] = {}
        for (sym, _cls), res in zip(self.universe, results):
            if isinstance(res, Exception):
                logger.warning("Análisis fallido para %s: %s", sym, res)
                continue
            rankings[sym] = res
        self.current_rankings = rankings
        self.historical_rankings.append(
            {"timestamp": datetime.now(), "scores": {s: a.final_score for s, a in rankings.items()}}
        )
        self._last_changes = self.detect_significant_changes(previous, rankings)
        return rankings

    async def analyze_single_asset(self, symbol: str, asset_class: str,
                                   ohlcv: pd.DataFrame | None = None,
                                   context: dict | None = None) -> AssetAnalysis:
        """Análisis completo de un activo. Acepta datos inyectados o vía provider."""
        if ohlcv is None:
            if self.data_provider is None:
                raise ValueError(f"Sin data_provider ni OHLCV para {symbol}")
            ohlcv = self.data_provider(symbol, asset_class)
            if asyncio.iscoroutine(ohlcv):
                ohlcv = await ohlcv
        if context is None and self.context_provider is not None:
            context = self.context_provider(symbol, asset_class)
            if asyncio.iscoroutine(context):
                context = await context
        context = context or {}

        df = self.technical_engine.calculate_all_indicators(ohlcv)
        last = df.iloc[-1]
        tech_sig = self.signal_generator.generate_signals(df)

        # Score técnico 0-100 a partir del confluence_score (-1..1).
        technical_score = float(np.clip(50 + 50 * tech_sig["confluence_score"], 0, 100))

        scores = {
            "technical": technical_score,
            "sentiment": float(context.get("sentiment_score", _NEUTRAL)),
            "macro": float(context.get("macro_score", _NEUTRAL)),
            "options_flow": float(context.get("options_flow_score", _NEUTRAL)),
            "ml_prediction": float(context.get("ml_prediction_score", _NEUTRAL)),
            "seasonality": float(context.get("seasonality_score", _NEUTRAL)),
            "volume_profile": float(context.get("volume_profile_score", _NEUTRAL)),
            "correlation": float(context.get("correlation_score", _NEUTRAL)),
        }
        final_score = float(sum(self.weights[k] * scores[k] for k in self.weights))
        action, strength = self._score_to_action(final_score)

        price = tech_sig["price"]
        atr = float(last.get("atr", price * 0.01)) or price * 0.01
        # Si la señal técnica no fija niveles (HOLD), los derivamos del ATR.
        entry = tech_sig.get("entry_price", price)
        stop = tech_sig.get("stop_loss", price - 1.5 * atr)
        tp1 = tech_sig.get("take_profit_1", price + 2 * atr)
        tp2 = tech_sig.get("take_profit_2", price + 3 * atr)
        tp3 = price + 4 * atr
        rr = tech_sig.get("risk_reward_ratio", 0.0) or (
            abs(tp1 - entry) / abs(entry - stop) if entry != stop else 0.0
        )

        vol = df["volume"]
        vol_anomaly = float(vol.iloc[-1] / vol.tail(20).mean()) if vol.tail(20).mean() else 1.0

        analysis = AssetAnalysis(
            symbol=symbol,
            asset_class=asset_class,
            current_price=price,
            timestamp=datetime.now(),
            technical_score=scores["technical"],
            sentiment_score=scores["sentiment"],
            macro_score=scores["macro"],
            options_flow_score=scores["options_flow"],
            ml_prediction_score=scores["ml_prediction"],
            seasonality_score=scores["seasonality"],
            volume_profile_score=scores["volume_profile"],
            correlation_score=scores["correlation"],
            final_score=final_score,
            signal_strength=strength,
            action=action,
            confidence=tech_sig["confidence"],
            expected_return_24h=context.get("expected_return_24h", (final_score - 50) / 10),
            expected_return_7d=context.get("expected_return_7d", (final_score - 50) / 4),
            risk_reward_ratio=rr,
            entry_price=entry,
            stop_loss=stop,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            technical_indicators={
                "rsi": float(last.get("rsi", np.nan)),
                "macd_diff": float(last.get("macd_diff", np.nan)),
                "atr": atr,
                "adx": float(last.get("adx", np.nan)),
            },
            sentiment_data=context.get("sentiment_data", {}),
            macro_events=context.get("macro_events", []),
            options_activity=context.get("options_activity", {}),
            recent_news=context.get("recent_news", []),
            social_mentions=int(context.get("social_mentions", 0)),
            volume_anomaly=vol_anomaly,
        )
        analysis.reasons_to_buy = self.generate_reasons_to_buy(analysis)
        analysis.reasons_to_avoid = self.generate_reasons_to_avoid(analysis)
        analysis.key_catalysts = self.identify_key_catalysts(analysis)
        analysis.risk_factors = self.identify_risk_factors(analysis)
        return analysis

    def _score_to_action(self, score: float) -> tuple[str, str]:
        t = self.thresholds
        if score >= 85:
            return "STRONG_BUY", "EXTREME_BUY"
        if score >= t["strong_buy"]:
            return "STRONG_BUY", "STRONG_BUY"
        if score >= t["buy"]:
            return "BUY", "BUY"
        if score <= t["strong_sell"]:
            return "STRONG_SELL", "STRONG_SELL"
        if score <= t["sell"]:
            return "SELL", "SELL"
        return "HOLD", "NEUTRAL"

    # ----------------------------------------------------------- explicaciones
    def generate_reasons_to_buy(self, a: AssetAnalysis) -> list[str]:
        r = []
        ind = a.technical_indicators
        if a.technical_score > 65:
            r.append("✅ Confluencia técnica alcista en múltiples categorías")
        if ind.get("rsi", 50) < 35:
            r.append(f"✅ RSI en zona de sobreventa ({ind['rsi']:.1f}) - oportunidad de entrada")
        if ind.get("macd_diff", 0) > 0:
            r.append("✅ MACD con histograma positivo - momentum alcista")
        if a.sentiment_score > 70:
            r.append(f"✅ Sentimiento del mercado muy positivo ({a.sentiment_score:.0f}/100)")
        if a.options_flow_score > 70:
            r.append("✅ Flujo de opciones alcista - smart money comprando")
        if a.ml_prediction_score > 65:
            r.append(f"✅ Modelo ML anticipa subida ({a.expected_return_24h:+.1f}% en 24h)")
        if a.volume_anomaly > 1.8:
            r.append(f"✅ Volumen {a.volume_anomaly:.1f}x sobre la media - interés institucional")
        if a.risk_reward_ratio >= 2.0:
            r.append(f"✅ Buen ratio riesgo/beneficio ({a.risk_reward_ratio:.1f})")
        return r

    def generate_reasons_to_avoid(self, a: AssetAnalysis) -> list[str]:
        r = []
        ind = a.technical_indicators
        if a.technical_score < 35:
            r.append("❌ Estructura técnica bajista - no luchar contra la tendencia")
        if ind.get("rsi", 50) > 75:
            r.append(f"❌ RSI en sobrecompra ({ind['rsi']:.1f}) - posible corrección")
        if a.sentiment_score < 30:
            r.append(f"❌ Sentimiento del mercado muy negativo ({a.sentiment_score:.0f}/100)")
        if 0 < a.risk_reward_ratio < 1.5:
            r.append(f"❌ Ratio riesgo/beneficio pobre ({a.risk_reward_ratio:.1f})")
        if a.correlation_score < 30:
            r.append("❌ Alta correlación con posiciones existentes - poca diversificación")
        return r

    def identify_key_catalysts(self, a: AssetAnalysis) -> list[str]:
        cats = []
        for ev in a.macro_events:
            cats.append(
                f"📅 {ev.get('event','Evento')} - {ev.get('date','')} (Impacto: {ev.get('impact','?')})"
            )
        return cats

    def identify_risk_factors(self, a: AssetAnalysis) -> list[str]:
        r = []
        ind = a.technical_indicators
        if a.technical_indicators.get("rsi", 50) > 78:
            r.append("⚠️ Sentimiento técnico eufórico - riesgo de reversión")
        if a.volume_anomaly < 0.5:
            r.append("⚠️ Volumen bajo - posible falta de liquidez")
        atr = ind.get("atr", 0)
        if atr and a.current_price and atr / a.current_price > 0.04:
            r.append("⚠️ Volatilidad (ATR) elevada - riesgo de slippage")
        return r

    # ------------------------------------------------------------- consultas
    def get_top_opportunities(self, n: int = 10, min_score: float = 70) -> list[AssetAnalysis]:
        ranked = sorted(self.current_rankings.values(), key=lambda x: x.final_score, reverse=True)
        return [a for a in ranked if a.final_score >= min_score][:n]

    def get_top_risks(self, n: int = 10, max_score: float = 30) -> list[AssetAnalysis]:
        ranked = sorted(self.current_rankings.values(), key=lambda x: x.final_score)
        return [a for a in ranked if a.final_score <= max_score][:n]

    def get_ranking_by_class(self, asset_class: str) -> list[AssetAnalysis]:
        items = [a for a in self.current_rankings.values() if a.asset_class == asset_class]
        return sorted(items, key=lambda x: x.final_score, reverse=True)

    def detect_significant_changes(self, previous: dict, current: dict,
                                   score_threshold: float | None = None) -> list[dict]:
        """Devuelve eventos de cambio relevantes para disparar alertas."""
        threshold = score_threshold or self.thresholds.get("score_change", 10)
        events = []
        for sym, cur in current.items():
            prev = previous.get(sym)
            if prev is None:
                if cur.action in ("STRONG_BUY", "STRONG_SELL"):
                    events.append({"type": f"NEW_{cur.action}", "symbol": sym, "analysis": cur})
                continue
            delta = cur.final_score - prev.final_score
            if abs(delta) >= threshold:
                events.append(
                    {
                        "type": "SCORE_CHANGE",
                        "symbol": sym,
                        "score_change": delta,
                        "previous": prev,
                        "current": cur,
                    }
                )
            elif cur.action != prev.action and cur.action in ("STRONG_BUY", "STRONG_SELL"):
                events.append({"type": f"NEW_{cur.action}", "symbol": sym, "analysis": cur})
        return events

    def get_market_summary(self) -> dict:
        rankings = self.current_rankings
        n = len(rankings)
        if n == 0:
            return {"total_assets_analyzed": 0, "market_sentiment": "NEUTRAL", "timestamp": datetime.now()}
        scores = [a.final_score for a in rankings.values()]
        bullish = sum(1 for a in rankings.values() if a.action in ("BUY", "STRONG_BUY"))
        bearish = sum(1 for a in rankings.values() if a.action in ("SELL", "STRONG_SELL"))
        neutral = n - bullish - bearish
        avg = float(np.mean(scores))

        by_class = {}
        classes = {c for _s, c in self.universe} or {a.asset_class for a in rankings.values()}
        for cls in classes:
            items = self.get_ranking_by_class(cls)
            if not items:
                continue
            cls_bull = sum(1 for a in items if a.action in ("BUY", "STRONG_BUY"))
            by_class[cls] = {
                "avg_score": float(np.mean([a.final_score for a in items])),
                "bullish_pct": cls_bull / len(items),
                "top": items[0].symbol,
            }

        sentiment = "BULLISH" if avg > 60 else "BEARISH" if avg < 40 else "NEUTRAL"
        return {
            "timestamp": datetime.now(),
            "total_assets_analyzed": n,
            "average_score": avg,
            "market_sentiment": sentiment,
            "bullish_assets": bullish,
            "bearish_assets": bearish,
            "neutral_assets": neutral,
            "top_opportunities": [a.to_row() for a in self.get_top_opportunities(5, min_score=0)],
            "top_risks": [a.to_row() for a in self.get_top_risks(5, max_score=100)],
            "by_asset_class": by_class,
        }
