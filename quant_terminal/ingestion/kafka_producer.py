"""Productor de Kafka para los topics del sistema.

``kafka-python`` se importa de forma perezosa. Topics:
market-data-ticks, economic-events, social-media-feed, news-headlines,
options-flow, signals, alerts.
"""

from __future__ import annotations

import json

TOPICS = (
    "market-data-ticks",
    "economic-events",
    "social-media-feed",
    "news-headlines",
    "options-flow",
    "signals",
    "alerts",
)


class KafkaDataProducer:
    def __init__(self, bootstrap_servers: list[str] | None = None) -> None:
        self.bootstrap_servers = bootstrap_servers or ["localhost:9092"]
        self._producer = None

    @property
    def producer(self):
        if self._producer is None:
            from kafka import KafkaProducer

            self._producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            )
        return self._producer

    def _send(self, topic: str, data: dict):
        return self.producer.send(topic, data)

    async def produce_market_data(self, tick_data: dict):
        return self._send("market-data-ticks", tick_data)

    async def produce_economic_event(self, event_data: dict):
        return self._send("economic-events", event_data)

    async def produce_social_media_post(self, post_data: dict):
        return self._send("social-media-feed", post_data)

    async def produce_news(self, news_data: dict):
        return self._send("news-headlines", news_data)

    async def produce_options_flow(self, flow_data: dict):
        return self._send("options-flow", flow_data)

    async def produce_signal(self, signal_data: dict):
        return self._send("signals", signal_data)
