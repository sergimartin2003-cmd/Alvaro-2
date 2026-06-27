"""Bot de Telegram para Jarvis: comandos + chat libre vía long-polling.

Comandos: /start, /help, /briefing, /ranking [n], /risks, /analyze SÍMBOLO,
/sentiment, /recommendations. Cualquier texto sin comando se enruta al asistente
conversacional.

El parsing de comandos y el dispatch (`handle`) son testeables sin red; la capa
de red (getUpdates / sendMessage) usa `requests` de forma perezosa.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

HELP_TEXT = (
    "🤖 <b>Comandos de Jarvis</b>\n"
    "/briefing — briefing diario completo\n"
    "/ranking [n] — top N oportunidades (por defecto 5)\n"
    "/risks — activos a evitar / advertencias\n"
    "/analyze SÍMBOLO — análisis de un activo (ej: /analyze AAPL)\n"
    "/sentiment — sentimiento general del mercado\n"
    "/recommendations — acciones recomendadas hoy\n"
    "/help — esta ayuda\n\n"
    "También puedes escribirme en lenguaje natural."
)

# Lista para registrar en Telegram (setMyCommands).
BOT_COMMANDS = [
    {"command": "briefing", "description": "Briefing diario completo"},
    {"command": "ranking", "description": "Top N oportunidades"},
    {"command": "risks", "description": "Activos a evitar / riesgos"},
    {"command": "analyze", "description": "Analizar un activo (ej: AAPL)"},
    {"command": "sentiment", "description": "Sentimiento del mercado"},
    {"command": "recommendations", "description": "Acciones recomendadas"},
    {"command": "help", "description": "Ayuda"},
]


class JarvisTelegramBot:
    def __init__(self, config: dict | None = None, assistant=None, daily_advisor=None,
                 ranking_engine=None, bot_token: str | None = None) -> None:
        self.config = config or {}
        self.assistant = assistant
        self.daily_advisor = daily_advisor
        self.ranking_engine = ranking_engine
        tg = (self.config.get("telegram") or {})
        self.bot_token = bot_token or tg.get("bot_token")
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.running = False
        self._offset = 0

    # ----------------------------------------------------------- parsing
    @staticmethod
    def parse_command(text: str) -> tuple[str | None, str]:
        """Devuelve (comando_sin_barra | None, argumentos). None si no es comando."""
        text = (text or "").strip()
        if not text.startswith("/"):
            return None, text
        parts = text[1:].split(maxsplit=1)
        cmd = parts[0].lower()
        if "@" in cmd:  # /analyze@MiBot -> analyze
            cmd = cmd.split("@", 1)[0]
        args = parts[1].strip() if len(parts) > 1 else ""
        return cmd, args

    # ----------------------------------------------------------- dispatch
    async def handle(self, text: str) -> str:
        """Procesa un mensaje (comando o texto libre) y devuelve la respuesta."""
        cmd, args = self.parse_command(text)
        if cmd is None:
            if self.assistant is not None:
                return await self.assistant.process_text_input(text)
            return "No tengo el asistente conectado."

        if cmd in ("start", "help"):
            return HELP_TEXT
        if cmd == "briefing":
            return await self._briefing()
        if cmd == "ranking":
            n = self._parse_int(args, default=5)
            return await self._call_handler("_handle_top_assets", {"n": n})
        if cmd == "risks":
            return await self._call_handler("_handle_risk_warnings", {})
        if cmd == "analyze":
            symbol = args.split()[0].upper() if args else None
            if not symbol:
                return "Uso: /analyze SÍMBOLO (ej: /analyze AAPL)"
            return await self._call_handler("_handle_analyze_asset", {"symbol": symbol})
        if cmd == "sentiment":
            return await self._call_handler("_handle_market_sentiment", {})
        if cmd in ("recommendations", "actions", "acciones"):
            return await self._call_handler("_handle_recommendations", {})
        return f"Comando desconocido: /{cmd}. Usa /help para ver los disponibles."

    async def _briefing(self) -> str:
        if self.daily_advisor is not None:
            b = await self.daily_advisor.generate_daily_briefing()
            return self.daily_advisor.format_telegram_briefing(b)
        return await self._call_handler("_handle_daily_briefing", {})

    async def _call_handler(self, name: str, params: dict) -> str:
        if self.assistant is None or not hasattr(self.assistant, name):
            return "Función no disponible."
        return await getattr(self.assistant, name)(params)

    @staticmethod
    def _parse_int(s: str, default: int) -> int:
        try:
            return int(s.split()[0])
        except (ValueError, IndexError):
            return default

    # ----------------------------------------------------------- red (lazy)
    def register_commands(self) -> None:
        import requests

        requests.post(f"{self.base_url}/setMyCommands", json={"commands": BOT_COMMANDS}, timeout=10)

    async def _send(self, chat_id, text: str) -> None:
        import requests

        # Telegram limita a 4096 chars por mensaje.
        for chunk in (text[i:i + 4000] for i in range(0, len(text), 4000)):
            requests.post(f"{self.base_url}/sendMessage",
                          data={"chat_id": chat_id, "text": chunk, "parse_mode": "HTML"}, timeout=15)

    async def _get_updates(self, timeout: int = 25) -> list[dict]:
        import requests

        resp = requests.get(f"{self.base_url}/getUpdates",
                            params={"offset": self._offset, "timeout": timeout},
                            timeout=timeout + 10)
        data = resp.json()
        return data.get("result", []) if data.get("ok") else []

    async def run_polling(self, poll_interval: int = 1) -> None:
        """Bucle principal de long-polling. Bloqueante hasta self.running=False."""
        self.running = True
        try:
            self.register_commands()
        except Exception as exc:  # pragma: no cover - depende de red
            logger.warning("No se pudieron registrar comandos: %s", exc)
        logger.info("Bot de Telegram en marcha (long-polling)")
        while self.running:
            try:
                updates = await self._get_updates()
                for upd in updates:
                    self._offset = upd["update_id"] + 1
                    msg = upd.get("message") or upd.get("channel_post")
                    if not msg or "text" not in msg:
                        continue
                    chat_id = msg["chat"]["id"]
                    try:
                        response = await self.handle(msg["text"])
                    except Exception as exc:  # pragma: no cover
                        logger.error("Error procesando '%s': %s", msg.get("text"), exc)
                        response = "Ha ocurrido un error procesando tu mensaje."
                    await self._send(chat_id, response)
            except Exception as exc:  # pragma: no cover - depende de red
                logger.error("Error en polling: %s", exc)
                await asyncio.sleep(5)
            await asyncio.sleep(poll_interval)
