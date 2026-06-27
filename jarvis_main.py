"""Orquestador del sistema Jarvis (Jarvis Módulo 3).

Integra el motor de ranking, el advisor diario, las alertas Telegram y el
asistente conversacional. Soporta tareas programadas (briefing matutino, resumen
de cierre) y un modo interactivo por texto o voz.

Uso:
    python jarvis_main.py [config/jarvis_config.yaml]
    python jarvis_main.py --once        # genera un briefing y termina
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import datetime, time, timedelta

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("jarvis")


def _expand_env(obj):
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj


class JarvisSystem:
    def __init__(self, config_path: str = "config/jarvis_config.yaml") -> None:
        self.config = self._load_config(config_path)
        self.running = False
        self._setup_components()

    def _load_config(self, config_path: str) -> dict:
        try:
            import yaml

            with open(config_path, "r", encoding="utf-8") as f:
                return _expand_env(yaml.safe_load(f))
        except FileNotFoundError:
            logger.warning("Config %s no encontrada; usando valores por defecto", config_path)
            return {}

    def _setup_components(self) -> None:
        from quant_terminal.alerts.telegram_alerts import TelegramAlertSystem
        from quant_terminal.analysis.real_time_ranking import RealTimeRankingEngine
        from quant_terminal.ingestion.market_data import MarketDataClient
        from quant_terminal.jarvis.assistant import JarvisAssistant
        from quant_terminal.jarvis.daily_advisor import DailyMarketAdvisor
        from quant_terminal.jarvis.llm import LLMClient
        from quant_terminal.jarvis.telegram_bot import JarvisTelegramBot

        md = self.config.get("market_data", {})
        self.market_data_client = MarketDataClient(
            polygon_api_key=md.get("polygon_api_key"),
            alpaca_api_key=md.get("alpaca_api_key"),
            alpaca_secret=md.get("alpaca_secret"),
        )
        self.ranking_engine = RealTimeRankingEngine(self.config, data_provider=self._data_provider)
        self.llm_client = LLMClient(model=self.config.get("llm", {}).get("model", "claude-opus-4-8"))
        self.telegram = TelegramAlertSystem(self.config.get("telegram", {}))
        self.daily_advisor = DailyMarketAdvisor(
            self.config, ranking_engine=self.ranking_engine, llm_client=self.llm_client,
            telegram_alerts=self.telegram,
        )
        self.assistant = JarvisAssistant(
            self.config, daily_advisor=self.daily_advisor, llm_client=self.llm_client,
            ranking_engine=self.ranking_engine,
        )
        self.bot = JarvisTelegramBot(
            self.config, assistant=self.assistant, daily_advisor=self.daily_advisor,
            ranking_engine=self.ranking_engine,
        )
        logger.info("Componentes de Jarvis inicializados (LLM disponible: %s)", self.llm_client.available)

    def _data_provider(self, symbol: str, asset_class: str):
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        return self.market_data_client.get_historical_data(symbol, "1Day", start, end)

    # ----------------------------------------------------------- tareas
    async def run(self) -> None:
        self.running = True
        logger.info("Iniciando Jarvis System...")
        tasks = [self._ranking_loop(), self._morning_briefing_scheduler(),
                 self._closing_summary_scheduler()]
        if self.bot.bot_token:
            tasks.append(self.bot.run_polling())
            logger.info("Bot de Telegram activado")
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _ranking_loop(self, interval: int = 300) -> None:
        while self.running:
            try:
                await self.ranking_engine.analyze_all_assets()
            except Exception as exc:
                logger.error("Error en ranking: %s", exc)
            await asyncio.sleep(interval)

    async def _morning_briefing_scheduler(self, hour: int = 8) -> None:
        while self.running:
            now = datetime.now()
            target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            if now > target:
                target += timedelta(days=1)
            await asyncio.sleep((target - now).total_seconds())
            try:
                briefing = await self.daily_advisor.generate_daily_briefing()
                if self.telegram.chat_ids:
                    await self.daily_advisor.send_daily_briefing_to_telegram(briefing)
                logger.info("Briefing matutino generado")
            except Exception as exc:
                logger.error("Error en briefing matutino: %s", exc)

    async def _closing_summary_scheduler(self, hour: int = 16, minute: int = 30) -> None:
        while self.running:
            now = datetime.now()
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if now > target:
                target += timedelta(days=1)
            await asyncio.sleep((target - now).total_seconds())
            try:
                if self.telegram.chat_ids:
                    await self.daily_advisor.send_closing_summary_to_telegram()
                logger.info("Resumen de cierre enviado")
            except Exception as exc:
                logger.error("Error en resumen de cierre: %s", exc)

    async def run_once(self) -> str:
        """Genera y devuelve un briefing una sola vez (sin bucle)."""
        await self.ranking_engine.analyze_all_assets()
        briefing = await self.daily_advisor.generate_daily_briefing()
        return self.daily_advisor.format_telegram_briefing(briefing)

    async def start_interactive_mode(self) -> None:
        print("\n🤖 Jarvis iniciado. Escribe 'salir' para terminar, 'voz' para modo voz.\n")
        loop = asyncio.get_event_loop()
        while True:
            user_input = (await loop.run_in_executor(None, input, "Tú: ")).strip()
            if user_input.lower() in ("salir", "exit", "quit"):
                print(f"¡Hasta luego, {self.assistant.user_name}!")
                break
            if user_input.lower() == "voz":
                response = await self.assistant.process_voice_input()
            else:
                response = await self.assistant.process_text_input(user_input)
            print(f"\nJarvis: {response}\n")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    config_path = args[0] if args else "config/jarvis_config.yaml"
    jarvis = JarvisSystem(config_path)
    if "--once" in sys.argv:
        print(asyncio.run(jarvis.run_once()))
        return
    if "--bot" in sys.argv:
        async def _run_bot():
            await jarvis.ranking_engine.analyze_all_assets()
            await jarvis.bot.run_polling()
        asyncio.run(_run_bot())
        return
    asyncio.run(jarvis.start_interactive_mode())


if __name__ == "__main__":
    main()
