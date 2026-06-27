"""Daily Market Advisor (Jarvis Módulo 1).

Genera un briefing diario completo combinando el RealTimeRankingEngine con
proveedores opcionales de noticias, macro y sentimiento social. El resumen en
lenguaje natural usa el LLM si está disponible, con un fallback de plantilla en
español para funcionar offline.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DailyRecommendation:
    date: str
    top_buys: list[dict]
    top_avoids: list[dict]
    market_summary: str
    key_news: list[dict] = field(default_factory=list)
    macro_events: list[dict] = field(default_factory=list)
    social_sentiment: dict = field(default_factory=dict)
    trump_tweets: list[dict] = field(default_factory=list)
    sector_analysis: dict = field(default_factory=dict)
    risk_warnings: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)


class DailyMarketAdvisor:
    """Compila recomendaciones diarias a partir del ranking y otros proveedores."""

    def __init__(self, config: dict | None = None, ranking_engine=None, llm_client=None,
                 news_provider=None, macro_provider=None, social_provider=None,
                 telegram_alerts=None) -> None:
        self.config = config or {}
        self.ranking_engine = ranking_engine
        self.llm_client = llm_client
        self.news_provider = news_provider
        self.macro_provider = macro_provider
        self.social_provider = social_provider
        self.telegram_alerts = telegram_alerts
        self.user_name = self.config.get("user_name", "Álvaro")

    async def generate_daily_briefing(self, date: str | None = None) -> DailyRecommendation:
        date = date or datetime.now().strftime("%Y-%m-%d")
        logger.info("Generando briefing diario para %s", date)

        top_buys = await self._get_top_buys(10)
        top_avoids = await self._get_top_avoids(10)
        key_news = await self._maybe(self.news_provider, "todays_news")
        macro_events = await self._maybe(self.macro_provider, "events_for_date", date)
        social = await self._maybe(self.social_provider, "comprehensive_sentiment", default={})
        trump = await self._maybe(self.social_provider, "political_tweets")
        risk_warnings = self._identify_risk_warnings(social, macro_events)
        action_items = self._generate_action_items(top_buys, top_avoids, risk_warnings)
        summary = self._generate_natural_summary(top_buys, top_avoids, key_news, macro_events, social)

        return DailyRecommendation(
            date=date, top_buys=top_buys, top_avoids=top_avoids, market_summary=summary,
            key_news=key_news or [], macro_events=macro_events or [], social_sentiment=social or {},
            trump_tweets=trump or [], risk_warnings=risk_warnings, action_items=action_items,
        )

    # ---------------------------------------------------------- ranking → dicts
    async def _get_top_buys(self, n: int = 10) -> list[dict]:
        if self.ranking_engine is None:
            return []
        opps = self.ranking_engine.get_top_opportunities(n, min_score=70)
        out = []
        for i, a in enumerate(opps, 1):
            out.append({
                "rank": i, "symbol": a.symbol, "asset_class": a.asset_class,
                "score": a.final_score, "signal": a.action, "confidence": a.confidence,
                "current_price": a.current_price, "expected_return_24h": a.expected_return_24h,
                "expected_return_7d": a.expected_return_7d, "risk_reward_ratio": a.risk_reward_ratio,
                "entry_price": a.entry_price, "stop_loss": a.stop_loss, "take_profit": a.take_profit_2,
                "reasons_to_buy": a.reasons_to_buy[:5], "reasons_to_avoid": a.reasons_to_avoid[:3],
                "key_catalysts": a.key_catalysts[:3], "risk_factors": a.risk_factors[:3],
                "technical_score": a.technical_score, "sentiment_score": a.sentiment_score,
                "macro_score": a.macro_score, "options_flow_score": a.options_flow_score,
                "ml_prediction_score": a.ml_prediction_score, "sector": a.asset_class,
                "volume_anomaly": a.volume_anomaly, "institutional_activity": a.volume_anomaly > 2.0,
            })
        return out

    async def _get_top_avoids(self, n: int = 10) -> list[dict]:
        if self.ranking_engine is None:
            return []
        risks = self.ranking_engine.get_top_risks(n, max_score=30)
        out = []
        for i, a in enumerate(risks, 1):
            out.append({
                "rank": i, "symbol": a.symbol, "asset_class": a.asset_class,
                "score": a.final_score, "signal": a.action, "confidence": a.confidence,
                "current_price": a.current_price, "expected_decline_24h": a.expected_return_24h,
                "reasons_to_avoid": a.reasons_to_avoid[:5], "risk_factors": a.risk_factors[:3],
                "technical_score": a.technical_score, "sentiment_score": a.sentiment_score,
            })
        return out

    @staticmethod
    async def _maybe(provider, method: str, *args, default=None):
        """Llama provider.method(*args) si existe; soporta sync y async."""
        if provider is None or not hasattr(provider, method):
            return default if default is not None else []
        try:
            res = getattr(provider, method)(*args)
            if asyncio.iscoroutine(res):
                res = await res
            return res
        except Exception as exc:  # pragma: no cover - depende del provider
            logger.warning("Provider %s.%s falló: %s", provider, method, exc)
            return default if default is not None else []

    # ----------------------------------------------------------- derivaciones
    def _identify_risk_warnings(self, social: dict, macro_events: list) -> list[str]:
        warnings = []
        if social:
            fg = social.get("fear_greed_index")
            if fg is not None and fg > 80:
                warnings.append(f"⚠️ Fear & Greed Index en {fg} - euforia, riesgo de corrección")
            if fg is not None and fg < 20:
                warnings.append(f"⚠️ Fear & Greed Index en {fg} - miedo extremo, posible suelo")
        for ev in (macro_events or []):
            if ev.get("impact") in ("High", "HIGH"):
                warnings.append(f"⚠️ {ev.get('event')} hoy - esperar antes de posiciones grandes")
        return warnings

    def _generate_action_items(self, top_buys, top_avoids, risk_warnings) -> list[str]:
        actions = []
        for b in top_buys[:3]:
            actions.append(
                f"✅ COMPRAR: {b['symbol']} a ${b['entry_price']:.2f}, stop ${b['stop_loss']:.2f}, "
                f"TP ${b['take_profit']:.2f} (posición ~{min(2.0 * b['confidence'], 3.0):.1f}% del portfolio)"
            )
        for a in top_avoids[:2]:
            reason = a["reasons_to_avoid"][0] if a["reasons_to_avoid"] else "tendencia bajista"
            actions.append(f"✅ VENDER/EVITAR: {a['symbol']} - {reason}")
        if any("Fear" in w for w in risk_warnings):
            actions.append("✅ AJUSTAR tamaño de posiciones por sentimiento extremo")
        return actions

    def _generate_natural_summary(self, top_buys, top_avoids, key_news, macro_events, social) -> str:
        if self.llm_client is not None and getattr(self.llm_client, "available", False):
            try:
                return self._llm_summary(top_buys, macro_events, social)
            except Exception as exc:  # pragma: no cover - depende de red
                logger.warning("Resumen LLM falló, usando plantilla: %s", exc)
        return self._template_summary(top_buys, top_avoids, macro_events, social)

    def _llm_summary(self, top_buys, macro_events, social) -> str:
        context = {
            "fecha": datetime.now().strftime("%Y-%m-%d"),
            "top_oportunidades": [{"symbol": b["symbol"], "score": b["score"], "signal": b["signal"]}
                                  for b in top_buys[:3]],
            "eventos_macro": [e.get("event") for e in (macro_events or [])[:3]],
            "sentimiento": social.get("overall_sentiment") if social else None,
        }
        system = (
            f"Eres Jarvis, un asistente de trading experto. Te diriges al usuario como "
            f"'{self.user_name}'. Responde en español, claro y conciso (máx 250 palabras), "
            "profesional pero accesible. No es asesoramiento financiero."
        )
        prompt = f"Genera el resumen de mercado de hoy a partir de este contexto:\n{context}"
        return self.llm_client.generate(prompt, system=system, max_tokens=600)

    def _template_summary(self, top_buys, top_avoids, macro_events, social) -> str:
        lines = [f"Buenos días, {self.user_name}. Resumen del mercado de hoy:"]
        if top_buys:
            names = ", ".join(f"{b['symbol']} ({b['score']:.0f})" for b in top_buys[:3])
            lines.append(f"Mejores oportunidades: {names}.")
        else:
            lines.append("No hay oportunidades con score alto en este momento.")
        if social and social.get("overall_sentiment"):
            lines.append(f"Sentimiento general del mercado: {social['overall_sentiment']}.")
        macro = [e.get("event") for e in (macro_events or []) if e.get("impact") in ("High", "HIGH")]
        if macro:
            lines.append(f"Atención a eventos macro de alto impacto: {', '.join(macro[:3])}.")
        if top_avoids:
            lines.append(f"Evitar: {', '.join(a['symbol'] for a in top_avoids[:3])}.")
        lines.append("Recuerda: gestión de riesgo estricta. Esto no es asesoramiento financiero.")
        return " ".join(lines)

    # ----------------------------------------------------------- Telegram
    def format_telegram_briefing(self, b: DailyRecommendation) -> str:
        out = [f"🌅 <b>BRIEFING DIARIO - {b.date}</b> 🌅", "", b.market_summary, "",
               "🔥 <b>TOP OPORTUNIDADES:</b>"]
        for buy in b.top_buys[:3]:
            out.append(
                f"\n<b>{buy['rank']}. {buy['symbol']}</b> (Score {buy['score']:.1f})\n"
                f"💰 ${buy['current_price']:.2f} | Entrada ${buy['entry_price']:.2f} "
                f"Stop ${buy['stop_loss']:.2f} TP ${buy['take_profit']:.2f}"
            )
        if b.risk_warnings:
            out += ["", "⚠️ <b>RIESGOS:</b>"] + [f"• {w}" for w in b.risk_warnings[:3]]
        if b.action_items:
            out += ["", "✅ <b>ACCIONES:</b>"] + [f"• {a}" for a in b.action_items[:4]]
        return "\n".join(out)

    async def send_daily_briefing_to_telegram(self, briefing: DailyRecommendation):
        if self.telegram_alerts is None:
            raise ValueError("No hay telegram_alerts configurado")
        return await self.telegram_alerts.send(self.format_telegram_briefing(briefing), priority="HIGH")
