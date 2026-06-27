"""Alertas de Telegram con formato rico, priorización y anti-spam.

Las funciones ``format_*`` son puras (devuelven el texto del mensaje) para poder
testearlas sin red. El envío real (``_post``) usa ``requests`` de forma perezosa
con reintentos y exponential backoff.
"""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger(__name__)

_PRIORITY = {"EXTREME": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}


def _pct(x: float) -> str:
    return f"{x:+.1f}%"


class TelegramAlertSystem:
    """Construye y envía alertas de trading a Telegram."""

    def __init__(self, config: dict) -> None:
        self.bot_token = config.get("bot_token")
        self.chat_ids = config.get("chat_ids") or [config.get("chat_id")]
        self.chat_ids = [c for c in self.chat_ids if c]
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        # Anti-spam / rate limiting.
        self.max_per_hour = config.get("max_alerts_per_hour", 20)
        self.min_interval_s = config.get("min_interval_seconds", 1)
        self.dedup_window_s = config.get("dedup_window_seconds", 3600)
        self._sent_times: list[float] = []
        self._last_per_symbol: dict[str, float] = {}
        self._last_send = 0.0

    # ----------------------------------------------------------- formateo
    @staticmethod
    def format_strong_buy_alert(a) -> str:
        lines = [
            "🔥 <b>NUEVA OPORTUNIDAD STRONG BUY</b> 🔥",
            "",
            f"<b>{a.symbol}</b> acaba de entrar en zona de COMPRA FUERTE",
            "",
            f"📊 Score: <b>{a.final_score:.1f}/100</b>",
            f"💪 Confianza: {a.confidence:.0%}",
            f"🎯 Retorno esperado 24h: {_pct(a.expected_return_24h)}",
            f"🎯 Retorno esperado 7d: {_pct(a.expected_return_7d)}",
            "",
            "✅ <b>RAZONES PARA COMPRAR:</b>",
            *(a.reasons_to_buy or ["✅ Confluencia alcista general"]),
        ]
        if a.reasons_to_avoid:
            lines += ["", "❌ <b>RAZONES DE PRECAUCIÓN:</b>", *a.reasons_to_avoid]
        if a.key_catalysts:
            lines += ["", "🔑 <b>CATALIZADORES CLAVE:</b>", *a.key_catalysts]
        if a.risk_factors:
            lines += ["", "⚠️ <b>FACTORES DE RIESGO:</b>", *a.risk_factors]
        lines += [
            "",
            "💰 <b>NIVELES DE TRADING:</b>",
            f"Entrada: ${a.entry_price:.2f}",
            f"Stop Loss: ${a.stop_loss:.2f}",
            f"TP1: ${a.take_profit_1:.2f}  TP2: ${a.take_profit_2:.2f}  TP3: ${a.take_profit_3:.2f}",
            f"Risk/Reward: {a.risk_reward_ratio:.1f}",
            "",
            "📈 <b>DESGLOSE:</b>",
            f"• Técnico: {a.technical_score:.0f}/100",
            f"• Sentimiento: {a.sentiment_score:.0f}/100",
            f"• Macro: {a.macro_score:.0f}/100",
            f"• Opciones: {a.options_flow_score:.0f}/100",
            f"• ML: {a.ml_prediction_score:.0f}/100",
            "",
            f"⏰ {a.timestamp:%Y-%m-%d %H:%M:%S}",
        ]
        return "\n".join(lines)

    @staticmethod
    def format_strong_sell_alert(a) -> str:
        lines = [
            "🛑 <b>ALERTA STRONG SELL</b> 🛑",
            "",
            f"<b>{a.symbol}</b> ha entrado en zona de VENTA FUERTE",
            "",
            f"📊 Score: <b>{a.final_score:.1f}/100</b>",
            f"💪 Confianza: {a.confidence:.0%}",
            "",
            "❌ <b>FACTORES NEGATIVOS:</b>",
            *(a.reasons_to_avoid or ["❌ Estructura bajista dominante"]),
        ]
        if a.risk_factors:
            lines += ["", "⚠️ <b>RIESGOS:</b>", *a.risk_factors]
        lines += ["", f"⏰ {a.timestamp:%Y-%m-%d %H:%M:%S}"]
        return "\n".join(lines)

    @staticmethod
    def format_significant_change_alert(symbol, previous, current, score_change) -> str:
        arrow = "📈 SUBIÓ" if score_change > 0 else "📉 BAJÓ"
        return "\n".join(
            [
                "🚨 <b>CAMBIO SIGNIFICATIVO DETECTADO</b> 🚨",
                "",
                f"<b>{symbol}</b> {arrow} {abs(score_change):.1f} puntos",
                "",
                f"Score: {previous.final_score:.1f} → <b>{current.final_score:.1f}</b>",
                f"Señal: {previous.action} → <b>{current.action}</b>",
                "",
                "💰 Niveles actualizados:",
                f"Entrada: ${current.entry_price:.2f}",
                f"Stop Loss: ${current.stop_loss:.2f}",
                f"Take Profit: ${current.take_profit_2:.2f}",
            ]
        )

    @staticmethod
    def format_daily_summary(summary: dict) -> str:
        n = summary.get("total_assets_analyzed", 0)
        lines = [
            "📊 <b>RESUMEN DIARIO DEL MERCADO</b> 📊",
            f"{summary['timestamp']:%Y-%m-%d %H:%M:%S}",
            "",
            f"🌍 SENTIMIENTO GENERAL: <b>{summary.get('market_sentiment','NEUTRAL')}</b>",
            f"📈 Alcistas: {summary.get('bullish_assets',0)}  "
            f"📉 Bajistas: {summary.get('bearish_assets',0)}  "
            f"⚖️ Neutrales: {summary.get('neutral_assets',0)}  (de {n})",
            "",
            "🔥 <b>TOP 5 OPORTUNIDADES:</b>",
        ]
        for i, o in enumerate(summary.get("top_opportunities", []), 1):
            lines.append(f"{i}. {o['symbol']} - Score: {o['score']} - {o['signal']}")
        lines += ["", "⚠️ <b>TOP 5 RIESGOS:</b>"]
        for i, o in enumerate(summary.get("top_risks", []), 1):
            lines.append(f"{i}. {o['symbol']} - Score: {o['score']} - {o['signal']}")
        if summary.get("by_asset_class"):
            lines += ["", "📊 <b>POR CLASE:</b>"]
            for cls, d in summary["by_asset_class"].items():
                lines.append(
                    f"• {cls.upper()}: {d['avg_score']:.1f}/100 "
                    f"({d['bullish_pct']:.0%} alcista) - Top: {d['top']}"
                )
        return "\n".join(lines)

    @staticmethod
    def format_economic_event_alert(event: dict, surprise_factor: float, market_impact: dict) -> str:
        lines = [
            "📅 <b>EVENTO ECONÓMICO PUBLICADO</b> 📅",
            "",
            f"<b>{event.get('event')}</b> ({event.get('country','')})",
            "",
            f"📊 Resultado: {event.get('actual')}",
            f"📊 Expectativa: {event.get('forecast')}",
            f"📊 Anterior: {event.get('previous')}",
            "",
            f"💥 Factor sorpresa: <b>{surprise_factor:+.2f}</b>",
            "",
            "📈 <b>IMPACTO EN MERCADOS:</b>",
        ]
        for asset, imp in market_impact.items():
            lines.append(f"• {asset}: {imp.get('direction')} ({_pct(imp.get('magnitude',0)*100)})")
        return "\n".join(lines)

    @staticmethod
    def format_trump_tweet_alert(tweet: dict, impact_analysis: dict) -> str:
        pa = impact_analysis.get("post_analysis", {})
        mip = impact_analysis.get("market_impact_prediction", {})
        lines = [
            "🐦 <b>TRUMP TWEET DETECTADO</b> 🐦",
            "",
            f"📝 {tweet.get('text','')}",
            "",
            "📊 <b>ANÁLISIS:</b>",
            f"• Sentimiento: {pa.get('sentiment')} ({pa.get('sentiment_score',0):+.2f})",
            f"• Tipos de política: {', '.join(pa.get('policy_types', [])) or 'N/A'}",
            f"• Agresividad: {pa.get('aggressiveness')}",
            f"• Score de impacto: {mip.get('impact_score',0):.2f}",
        ]
        if mip.get("tickers_mentioned"):
            lines.append(f"• Tickers: {', '.join(mip['tickers_mentioned'])}")
        return "\n".join(lines)

    @staticmethod
    def format_options_flow_alert(symbol: str, unusual: dict) -> str:
        return "\n".join(
            [
                "💰 <b>FLUJO INUSUAL DE OPCIONES</b> 💰",
                "",
                f"<b>{symbol}</b> - {unusual.get('direction','')} inusual",
                "",
                f"• Strike: ${unusual.get('strike')}",
                f"• Expiry: {unusual.get('expiry')}",
                f"• Volume: {unusual.get('volume'):,}",
                f"• Open Interest: {unusual.get('oi'):,}",
                f"• Ratio Vol/OI: {unusual.get('ratio',0):.1f}x",
            ]
        )

    # ----------------------------------------------------------- envío
    def _build_keyboard(self, symbol: str) -> dict:
        return {
            "inline_keyboard": [
                [
                    {"text": "📊 Ver gráfico", "url": f"https://www.tradingview.com/symbols/{symbol}/"},
                    {"text": "🔕 Silenciar", "callback_data": f"mute:{symbol}"},
                ]
            ]
        }

    def _allowed(self, symbol: str | None) -> bool:
        now = time.time()
        self._sent_times = [t for t in self._sent_times if now - t < 3600]
        if len(self._sent_times) >= self.max_per_hour:
            return False
        if now - self._last_send < self.min_interval_s:
            return False
        if symbol is not None:
            last = self._last_per_symbol.get(symbol, 0)
            if now - last < self.dedup_window_s:
                return False
        return True

    async def send(self, text: str, symbol: str | None = None, priority: str = "MEDIUM",
                   keyboard_symbol: str | None = None) -> dict:
        """Envía respetando rate limiting; EXTREME ignora la deduplicación."""
        if priority != "EXTREME" and not self._allowed(symbol):
            logger.info("Alerta suprimida por rate limiting: %s", symbol)
            return {"sent": False, "reason": "rate_limited"}

        now = time.time()
        self._sent_times.append(now)
        self._last_send = now
        if symbol is not None:
            self._last_per_symbol[symbol] = now

        results = {}
        for chat_id in self.chat_ids:
            reply_markup = self._build_keyboard(keyboard_symbol) if keyboard_symbol else None
            results[chat_id] = await self._post(chat_id, text, reply_markup)
        return {"sent": True, "results": results}

    async def _post(self, chat_id, text: str, reply_markup: dict | None,
                    retries: int = 3) -> dict:
        import json

        import requests

        payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
        if reply_markup:
            payload["reply_markup"] = json.dumps(reply_markup)
        delay = 1
        for attempt in range(retries):
            try:
                r = requests.post(f"{self.base_url}/sendMessage", data=payload, timeout=10)
                if r.status_code == 200:
                    return {"success": True}
                logger.warning("Telegram %s: %s", r.status_code, r.text[:120])
            except Exception as exc:  # pragma: no cover - depende de red
                logger.warning("Error enviando a Telegram: %s", exc)
            await asyncio.sleep(delay)
            delay *= 2
        return {"success": False}

    # ----------------------------------------------------- helpers de alto nivel
    async def send_strong_buy_alert(self, analysis):
        return await self.send(
            self.format_strong_buy_alert(analysis), symbol=analysis.symbol,
            priority="EXTREME" if analysis.final_score > 85 else "HIGH",
            keyboard_symbol=analysis.symbol,
        )

    async def send_strong_sell_alert(self, analysis):
        return await self.send(
            self.format_strong_sell_alert(analysis), symbol=analysis.symbol, priority="HIGH",
        )

    async def send_significant_change_alert(self, symbol, previous, current, score_change):
        return await self.send(
            self.format_significant_change_alert(symbol, previous, current, score_change),
            symbol=symbol, priority="HIGH",
        )

    async def send_daily_summary(self, ranking_engine):
        return await self.send(self.format_daily_summary(ranking_engine.get_market_summary()),
                               priority="MEDIUM")

    async def dispatch_change_events(self, events: list[dict]) -> list[dict]:
        """Convierte eventos del ranking engine en alertas, ordenados por prioridad."""
        out = []
        for ev in sorted(events, key=lambda e: _PRIORITY.get(self._event_priority(e), 0), reverse=True):
            t = ev["type"]
            if t == "NEW_STRONG_BUY":
                out.append(await self.send_strong_buy_alert(ev["analysis"]))
            elif t == "NEW_STRONG_SELL":
                out.append(await self.send_strong_sell_alert(ev["analysis"]))
            elif t == "SCORE_CHANGE":
                out.append(
                    await self.send_significant_change_alert(
                        ev["symbol"], ev["previous"], ev["current"], ev["score_change"]
                    )
                )
        return out

    @staticmethod
    def _event_priority(ev: dict) -> str:
        if ev["type"] == "NEW_STRONG_BUY":
            a = ev.get("analysis")
            return "EXTREME" if a and a.final_score > 85 else "HIGH"
        if ev["type"] == "NEW_STRONG_SELL":
            return "HIGH"
        return "MEDIUM"
