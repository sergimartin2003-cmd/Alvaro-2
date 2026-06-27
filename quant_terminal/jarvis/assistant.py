"""Jarvis Conversational Assistant (Jarvis Módulo 2).

Interfaz conversacional: detecta intención (reglas, testeable), ejecuta comandos
contra el DailyMarketAdvisor / ranking engine y responde en lenguaje natural.
La voz (speech_recognition / gTTS / pygame) se importa de forma perezosa.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)

_INTENT_PATTERNS = {
    "daily_briefing": ["briefing diario", "resumen del día", "dame el briefing", "qué pasa hoy"],
    "top_assets": ["mejores activos", "top activos", "activos", "qué debo comprar",
                   "oportunidades", "recomendaciones"],
    "news": ["noticias", "qué noticias", "headlines", "qué está pasando"],
    "macro_events": ["eventos macro", "eventos económicos", "calendario económico", "fomc", "fed"],
    "analyze_asset": ["analiza", "analizar", "qué opinas de", "dime sobre"],
    "market_sentiment": ["sentimiento", "cómo está el mercado", "fear greed"],
    "trump_tweets": ["trump", "qué dice trump", "tweets de trump", "geopolítico"],
    "risk_warnings": ["riesgos", "advertencias", "precaución", "cuidado"],
    "recommendations": ["qué hago", "qué recomiendas", "qué debo hacer", "acciones"],
    "help": ["ayuda", "help", "qué puedes hacer", "comandos"],
}


class JarvisAssistant:
    def __init__(self, config: dict | None = None, daily_advisor=None, llm_client=None,
                 ranking_engine=None) -> None:
        self.config = config or {}
        self.daily_advisor = daily_advisor
        self.llm_client = llm_client
        self.ranking_engine = ranking_engine
        self.conversation_history: list[dict] = []
        self.user_name = self.config.get("user_name", "Álvaro")
        self.commands = {
            "daily_briefing": self._handle_daily_briefing,
            "top_assets": self._handle_top_assets,
            "analyze_asset": self._handle_analyze_asset,
            "market_sentiment": self._handle_market_sentiment,
            "risk_warnings": self._handle_risk_warnings,
            "recommendations": self._handle_recommendations,
            "help": self._handle_help,
        }

    # ----------------------------------------------------------- intención
    def detect_intent(self, user_input: str) -> dict:
        low = user_input.lower()
        for command, patterns in _INTENT_PATTERNS.items():
            for p in patterns:
                if p in low:
                    return {"command": command, "parameters": self._extract_parameters(command, user_input),
                            "confidence": 0.95}
        return {"command": "general", "parameters": {"query": user_input}, "confidence": 0.5}

    @staticmethod
    def _extract_parameters(command: str, user_input: str) -> dict:
        params: dict = {}
        if command == "analyze_asset":
            for word in user_input.replace("/", " ").split():
                token = word.strip(".,¿?¡!").upper()
                if token.isalpha() and 1 < len(token) <= 5 and token == word.strip(".,¿?¡!").upper() \
                        and word.strip(".,¿?¡!").isupper():
                    params["symbol"] = token
                    break
            if "symbol" not in params:
                # último token alfanumérico en mayúsculas tipo ticker
                m = re.findall(r"\b[A-Z]{2,5}\b", user_input)
                if m:
                    params["symbol"] = m[-1]
        elif command == "top_assets":
            nums = re.findall(r"\d+", user_input)
            params["n"] = int(nums[0]) if nums else 10
        return params

    # ----------------------------------------------------------- proceso
    async def process_text_input(self, user_input: str) -> str:
        self.conversation_history.append({"role": "user", "content": user_input,
                                          "timestamp": datetime.now().isoformat()})
        intent = self.detect_intent(user_input)
        handler = self.commands.get(intent["command"])
        if handler is not None:
            response = await handler(intent["parameters"])
        else:
            response = await self._handle_general_query(user_input)
        self.conversation_history.append({"role": "assistant", "content": response,
                                          "timestamp": datetime.now().isoformat()})
        return response

    async def process_voice_input(self) -> str:
        import speech_recognition as sr  # lazy

        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        try:
            text = recognizer.recognize_google(audio, language="es-ES")
        except Exception:
            return "No he podido entenderte, ¿puedes repetir?"
        response = await self.process_text_input(text)
        try:
            self.speak(response)
        except Exception as exc:  # pragma: no cover
            logger.warning("TTS falló: %s", exc)
        return response

    def speak(self, text: str) -> None:
        import io

        import pygame
        from gtts import gTTS

        buf = io.BytesIO()
        gTTS(text=text, lang="es", slow=False).write_to_fp(buf)
        buf.seek(0)
        pygame.mixer.init()
        pygame.mixer.music.load(buf)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    # ----------------------------------------------------------- handlers
    async def _handle_daily_briefing(self, params: dict) -> str:
        if self.daily_advisor is None:
            return "No tengo el advisor diario configurado."
        b = await self.daily_advisor.generate_daily_briefing()
        out = [b.market_summary, "", "🔥 TOP OPORTUNIDADES:"]
        for buy in b.top_buys[:3]:
            out.append(f"{buy['rank']}. {buy['symbol']} (score {buy['score']:.1f}) — "
                       f"entrada ${buy['entry_price']:.2f}, stop ${buy['stop_loss']:.2f}, "
                       f"TP ${buy['take_profit']:.2f}")
        if b.action_items:
            out += ["", "✅ ACCIONES:"] + [f"• {a}" for a in b.action_items[:3]]
        return "\n".join(out)

    async def _handle_top_assets(self, params: dict) -> str:
        if self.daily_advisor is None:
            return "No tengo datos de ranking."
        n = params.get("n", 10)
        buys = await self.daily_advisor._get_top_buys(n)
        if not buys:
            return "No hay oportunidades con score suficiente ahora mismo."
        out = [f"🔥 TOP {len(buys)} OPORTUNIDADES:"]
        for b in buys:
            reason = b["reasons_to_buy"][0] if b["reasons_to_buy"] else ""
            out.append(f"{b['rank']}. {b['symbol']} — score {b['score']:.1f} ({b['signal']}), "
                       f"${b['current_price']:.2f}. {reason}")
        return "\n".join(out)

    async def _handle_analyze_asset(self, params: dict) -> str:
        symbol = params.get("symbol")
        if not symbol:
            return "¿Qué activo quieres que analice? Dime el símbolo (ej: AAPL)."
        if self.ranking_engine is None or symbol not in self.ranking_engine.current_rankings:
            return f"No tengo datos de {symbol} ahora mismo."
        a = self.ranking_engine.current_rankings[symbol]
        out = [
            f"📊 {symbol}: {a.action} (score {a.final_score:.1f}/100, confianza {a.confidence:.0%})",
            f"💰 Precio ${a.current_price:.2f} | Entrada ${a.entry_price:.2f} | "
            f"Stop ${a.stop_loss:.2f} | TP ${a.take_profit_2:.2f} | R/R {a.risk_reward_ratio:.1f}",
            "✅ Razones para comprar:" if a.reasons_to_buy else "Sin razones alcistas claras.",
        ]
        out += [f"  {r}" for r in a.reasons_to_buy[:4]]
        if a.reasons_to_avoid:
            out += ["❌ Precauciones:"] + [f"  {r}" for r in a.reasons_to_avoid[:3]]
        out.append(f"Desglose: téc {a.technical_score:.0f} / sent {a.sentiment_score:.0f} / "
                   f"macro {a.macro_score:.0f} / opc {a.options_flow_score:.0f} / ML {a.ml_prediction_score:.0f}")
        return "\n".join(out)

    async def _handle_market_sentiment(self, params: dict) -> str:
        if self.ranking_engine is None:
            return "No tengo datos de mercado."
        s = self.ranking_engine.get_market_summary()
        return (f"📊 Sentimiento: {s.get('market_sentiment', 'NEUTRAL')} "
                f"(score medio {s.get('average_score', 50):.1f}). "
                f"Alcistas {s.get('bullish_assets', 0)} / bajistas {s.get('bearish_assets', 0)} / "
                f"neutrales {s.get('neutral_assets', 0)} de {s.get('total_assets_analyzed', 0)}.")

    async def _handle_risk_warnings(self, params: dict) -> str:
        if self.daily_advisor is None:
            return "No tengo el advisor configurado."
        b = await self.daily_advisor.generate_daily_briefing()
        if not b.risk_warnings:
            return "No detecto riesgos relevantes ahora mismo."
        return "⚠️ Advertencias:\n" + "\n".join(f"• {w}" for w in b.risk_warnings)

    async def _handle_recommendations(self, params: dict) -> str:
        if self.daily_advisor is None:
            return "No tengo el advisor configurado."
        b = await self.daily_advisor.generate_daily_briefing()
        if not b.action_items:
            return "Sin acciones recomendadas claras ahora mismo."
        return "✅ Acciones recomendadas:\n" + "\n".join(f"• {a}" for a in b.action_items)

    async def _handle_help(self, params: dict) -> str:
        return (
            "🤖 Puedo ayudarte con:\n"
            "• 'Dame el briefing diario'\n"
            "• 'Dime los mejores activos' / 'top 5 activos'\n"
            "• 'Analiza AAPL'\n"
            "• 'Sentimiento del mercado'\n"
            "• '¿Qué riesgos hay?'\n"
            "• '¿Qué debo hacer?'\n"
            "También puedes preguntarme en lenguaje natural."
        )

    async def _handle_general_query(self, user_input: str) -> str:
        if self.llm_client is not None and getattr(self.llm_client, "available", False):
            system = (f"Eres Jarvis, asistente de trading. Te diriges al usuario como "
                      f"'{self.user_name}'. Responde en español, conciso y accionable. "
                      "No es asesoramiento financiero.")
            history = "\n".join(f"{m['role']}: {m['content']}" for m in self.conversation_history[-6:])
            try:
                return self.llm_client.generate(f"Historial:\n{history}\n\nPregunta: {user_input}",
                                                system=system, max_tokens=500)
            except Exception as exc:  # pragma: no cover
                logger.warning("LLM general falló: %s", exc)
        return ("No tengo un LLM conectado para responder eso libremente. Prueba con "
                "'ayuda' para ver lo que sí puedo hacer.")
