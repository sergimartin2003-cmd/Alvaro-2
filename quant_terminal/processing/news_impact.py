"""Predicción de impacto de noticias financieras en el mercado."""

from __future__ import annotations

from .sentiment import FinancialSentimentAnalyzer

_NEWS_TYPES = {
    "rate_hike": ["rate hike", "raises rates", "tightening"],
    "rate_cut": ["rate cut", "cuts rates", "easing"],
    "earnings_beat": ["beats earnings", "tops estimates", "earnings beat"],
    "earnings_miss": ["misses earnings", "below estimates", "earnings miss"],
    "merger": ["merger", "acquisition", "to acquire", "buyout"],
}

_HISTORICAL_IMPACTS = {
    "rate_hike": {"USD": ("POSITIVE", 0.02), "BONDS": ("NEGATIVE", -0.01), "EQUITIES": ("NEGATIVE", -0.015)},
    "rate_cut": {"USD": ("NEGATIVE", -0.02), "BONDS": ("POSITIVE", 0.01), "EQUITIES": ("POSITIVE", 0.015)},
    "earnings_beat": {"EQUITIES": ("POSITIVE", 0.035)},
    "earnings_miss": {"EQUITIES": ("NEGATIVE", -0.04)},
    "merger": {"EQUITIES": ("POSITIVE", 0.05)},
}


class NewsImpactAnalyzer:
    """Clasifica noticias y estima impacto usando una base histórica simple."""

    def __init__(self, sentiment_analyzer: FinancialSentimentAnalyzer | None = None) -> None:
        self.sentiment_analyzer = sentiment_analyzer or FinancialSentimentAnalyzer()
        self.historical_impacts = _HISTORICAL_IMPACTS

    def _classify(self, text: str) -> str | None:
        low = text.lower()
        for ntype, keywords in _NEWS_TYPES.items():
            if any(k in low for k in keywords):
                return ntype
        return None

    def predict_market_impact(self, news: dict) -> dict:
        text = f"{news.get('headline', '')} {news.get('body', '')}"
        sentiment = self.sentiment_analyzer.analyze_text(text, source="news")
        ntype = self._classify(text)
        entities = self.sentiment_analyzer.extract_financial_entities(text)

        predicted = {}
        if ntype and ntype in self.historical_impacts:
            for asset, (direction, magnitude) in self.historical_impacts[ntype].items():
                predicted[asset] = {
                    "direction": direction,
                    "estimated_move": magnitude,
                    "confidence": 0.7,
                    "duration": "hours",
                }

        return {
            "news_classification": {
                "type": (ntype or "GENERAL").upper(),
                "importance": "HIGH" if ntype else "LOW",
                "sentiment": sentiment["ensemble_sentiment"]["label"].upper(),
            },
            "predicted_impact": predicted,
            "tickers_mentioned": entities["tickers"],
            "sentiment": sentiment["ensemble_sentiment"],
        }

    @staticmethod
    def calculate_surprise_factor(news: dict) -> float:
        actual = news.get("actual")
        forecast = news.get("forecast")
        std = news.get("std_dev") or 1.0
        if actual is None or forecast is None:
            return 0.0
        return (actual - forecast) / std
