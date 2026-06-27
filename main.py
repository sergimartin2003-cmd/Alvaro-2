"""Orquestador principal de la terminal de trading.

Integra ingesta, ranking en tiempo real, alertas de Telegram y dashboard web,
y los ejecuta en paralelo con asyncio. Apagado ordenado vía señales del SO.

Uso:
    python main.py [config/config.yaml]

Requiere las dependencias de ``requirements-optional.txt`` para la operación en
vivo (Kafka, market data, Telegram, Dash). El ranking engine puede correr con un
``data_provider`` propio para datos históricos/sintéticos.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("orchestrator")


def _expand_env(obj):
    """Sustituye ${VAR} por variables de entorno recursivamente."""
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj


class TradingSystemOrchestrator:
    def __init__(self, config_path: str = "config/config.yaml") -> None:
        self.config = self._load_config(config_path)
        self.running = False
        self._setup_components()

    def _load_config(self, config_path: str) -> dict:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            return _expand_env(yaml.safe_load(f))

    def _setup_components(self) -> None:
        from quant_terminal.alerts.telegram_alerts import TelegramAlertSystem
        from quant_terminal.analysis.real_time_ranking import RealTimeRankingEngine
        from quant_terminal.ingestion.market_data import MarketDataClient

        md = self.config.get("market_data", {})
        self.market_data_client = MarketDataClient(
            polygon_api_key=md.get("polygon_api_key"),
            alpaca_api_key=md.get("alpaca_api_key"),
            alpaca_secret=md.get("alpaca_secret"),
        )
        self.ranking_engine = RealTimeRankingEngine(
            self.config, data_provider=self._default_data_provider
        )
        self.telegram_alerts = TelegramAlertSystem(self.config.get("telegram", {}))

        # El dashboard es opcional (depende de Dash).
        self.dashboard = None
        try:
            from quant_terminal.dashboard.main_dashboard import TradingDashboard

            self.dashboard = TradingDashboard(self.ranking_engine, self.telegram_alerts)
        except Exception as exc:  # pragma: no cover
            logger.warning("Dashboard no disponible: %s", exc)
        logger.info("Componentes inicializados")

    def _default_data_provider(self, symbol: str, asset_class: str):
        """Datos históricos recientes vía Alpaca (timeframe diario, ~1 año)."""
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        return self.market_data_client.get_historical_data(symbol, "1Day", start, end)

    # --------------------------------------------------------------- tareas
    async def run(self) -> None:
        self.running = True
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
            except NotImplementedError:  # pragma: no cover - Windows
                pass

        logger.info("Iniciando Trading System Orchestrator...")
        tasks = [
            self.run_ranking_engine(),
            self.run_daily_summary_scheduler(),
            self.run_health_monitoring(),
        ]
        if self.dashboard is not None:
            tasks.append(self.run_dashboard())
        await asyncio.gather(*tasks, return_exceptions=True)

    async def run_ranking_engine(self) -> None:
        logger.info("Ranking engine: análisis completo cada 5 min, top10 cada 30 s")
        last_full = 0.0
        last_top10 = 0.0
        while self.running:
            try:
                now = datetime.now().timestamp()
                if now - last_full >= self.config.get("ranking_engine", {}).get("update_interval_full", 300):
                    await self.ranking_engine.analyze_all_assets()
                    events = getattr(self.ranking_engine, "_last_changes", [])
                    if events:
                        await self.telegram_alerts.dispatch_change_events(events)
                    last_full = now
                elif now - last_top10 >= self.config.get("ranking_engine", {}).get("update_interval_top10", 30):
                    for a in self.ranking_engine.get_top_opportunities(10, min_score=0):
                        try:
                            self.ranking_engine.current_rankings[a.symbol] = (
                                await self.ranking_engine.analyze_single_asset(a.symbol, a.asset_class)
                            )
                        except Exception as exc:
                            logger.warning("Update top10 %s: %s", a.symbol, exc)
                    last_top10 = now
                await asyncio.sleep(1)
            except Exception as exc:
                logger.error("Error en ranking engine: %s", exc)
                await asyncio.sleep(10)

    async def run_dashboard(self) -> None:
        import threading

        cfg = self.config.get("dashboard", {})
        logger.info("Dashboard en http://%s:%s", cfg.get("host", "0.0.0.0"), cfg.get("port", 8050))
        thread = threading.Thread(
            target=self.dashboard.run,
            kwargs={"host": cfg.get("host", "0.0.0.0"), "port": cfg.get("port", 8050)},
            daemon=True,
        )
        thread.start()
        while self.running:
            await asyncio.sleep(1)

    async def run_health_monitoring(self, interval: int = 60) -> None:
        while self.running:
            health = self.check_system_health()
            if not health["healthy"]:
                logger.warning("Problemas de salud: %s", health["issues"])
            await asyncio.sleep(interval)

    def check_system_health(self) -> dict:
        issues = []
        if not self.ranking_engine.current_rankings:
            issues.append("ranking vacío")
        return {"healthy": not issues, "issues": issues}

    async def run_daily_summary_scheduler(self, hour: int = 9) -> None:
        while self.running:
            try:
                now = datetime.now()
                target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
                if now > target:
                    target += timedelta(days=1)
                wait = (target - now).total_seconds()
                logger.info("Próximo resumen diario en %.1f h", wait / 3600)
                await asyncio.sleep(wait)
                await self.telegram_alerts.send_daily_summary(self.ranking_engine)
            except Exception as exc:
                logger.error("Error en resumen diario: %s", exc)
                await asyncio.sleep(3600)

    async def shutdown(self) -> None:
        logger.info("Apagando orquestador...")
        self.running = False
        await asyncio.sleep(0.1)


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    orchestrator = TradingSystemOrchestrator(config_path)
    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        logger.info("Interrumpido por el usuario")


if __name__ == "__main__":
    main()
