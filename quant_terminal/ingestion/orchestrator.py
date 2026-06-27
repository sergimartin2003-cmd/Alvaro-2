"""Orquestador del pipeline de ingesta: corre todas las fuentes en paralelo."""

from __future__ import annotations

import asyncio

from .forex_factory import ForexFactoryScraper
from .kafka_producer import KafkaDataProducer
from .market_data import MarketDataClient
from .news_aggregator import NewsAggregator


class DataPipelineOrchestrator:
    """Coordina los productores de datos hacia Kafka.

    Cada tarea está aislada para que un fallo en una fuente no detenga el resto.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.kafka_producer = KafkaDataProducer(config.get("kafka", {}).get("bootstrap_servers"))
        self.forex_scraper = ForexFactoryScraper()
        self.news_aggregator = NewsAggregator()
        md = config.get("market_data", {})
        self.market_data_client = MarketDataClient(
            polygon_api_key=md.get("polygon_api_key"),
            alpaca_api_key=md.get("alpaca_api_key"),
            alpaca_secret=md.get("alpaca_secret"),
        )
        self.symbols = config.get("symbols", ["SPY", "QQQ"])

    async def run(self) -> None:
        await asyncio.gather(
            self._market_data_stream(),
            self._economic_events_check(),
            self._news_fetch(),
            return_exceptions=True,
        )

    async def _market_data_stream(self) -> None:
        async def _on_quote(q):
            await self.kafka_producer.produce_market_data(
                {"symbol": getattr(q, "symbol", None), "bid": getattr(q, "bid_price", None),
                 "ask": getattr(q, "ask_price", None)}
            )

        await self.market_data_client.stream_quotes(self.symbols, _on_quote)

    async def _economic_events_check(self, interval: int = 60) -> None:
        while True:
            try:
                events = self.forex_scraper.get_events("today")
                for _, event in events.iterrows():
                    await self.kafka_producer.produce_economic_event(event.to_dict())
            except Exception:
                pass
            await asyncio.sleep(interval)

    async def _news_fetch(self, interval: int = 30) -> None:
        while True:
            try:
                news = self.news_aggregator.fetch_all_feeds()
                analysis = self.news_aggregator.analyze_news_sentiment(
                    self.news_aggregator.filter_market_moving(news)
                )
                for item in analysis["market_moving_news"]:
                    await self.kafka_producer.produce_news(item)
            except Exception:
                pass
            await asyncio.sleep(interval)
