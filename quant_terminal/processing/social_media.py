"""Análisis de redes sociales (cuentas políticas / influencers) e impacto."""

from __future__ import annotations

from .sentiment import FinancialSentimentAnalyzer

_POLICY_KEYWORDS = {
    "TRADE_POLICY": ["tariff", "tariffs", "trade war", "sanctions", "import", "export"],
    "MONETARY_POLICY": ["fed", "rate", "powell", "interest rate"],
    "FISCAL_POLICY": ["tax", "spending", "budget", "stimulus"],
    "REGULATION": ["regulation", "antitrust", "ban", "investigation"],
}


class SocialMediaAnalyzer:
    """Sentimiento e impacto de posts de cuentas de alto alcance."""

    def __init__(self, sentiment_analyzer: FinancialSentimentAnalyzer | None = None) -> None:
        self.sentiment_analyzer = sentiment_analyzer or FinancialSentimentAnalyzer()
        self.influencer_accounts = {
            "trump": {"id": "25073877", "name": "realDonaldTrump", "influence": 0.95},
            "elon": {"id": "44196397", "name": "elonmusk", "influence": 0.90},
            "potus": {"id": "822215679726100480", "name": "POTUS", "influence": 0.85},
        }

    def _detect_policy(self, text: str) -> list[str]:
        low = text.lower()
        return [p for p, kws in _POLICY_KEYWORDS.items() if any(k in low for k in kws)]

    def _influence(self, author_id: str) -> float:
        for data in self.influencer_accounts.values():
            if data["id"] == author_id:
                return data["influence"]
        return 0.3

    def analyze_political_post(self, post: dict) -> dict:
        text = post.get("text", "")
        sentiment = self.sentiment_analyzer.analyze_text(text, source="twitter")
        entities = self.sentiment_analyzer.extract_financial_entities(text)
        policies = self._detect_policy(text)
        influence = self._influence(post.get("author_id", ""))
        score = sentiment["ensemble_sentiment"]["score"]
        impact = min(1.0, abs(score) * influence + 0.3 * len(policies))

        return {
            "post_analysis": {
                "sentiment": sentiment["ensemble_sentiment"]["label"].upper(),
                "sentiment_score": score,
                "policy_types": policies,
                "aggressiveness": "HIGH" if abs(score) > 0.6 else "MEDIUM",
            },
            "market_impact_prediction": {
                "impact_score": impact,
                "tickers_mentioned": entities["tickers"],
            },
            "alert_priority": "HIGH" if impact > 0.7 else "MEDIUM" if impact > 0.4 else "LOW",
        }

    def track_sentiment_momentum(self, posts: list[dict], window_hours: int = 24) -> dict:
        if not posts:
            return {"current_sentiment": 0.0, "sentiment_trend": "STABLE"}
        scores = [
            self.sentiment_analyzer.analyze_text(p.get("text", ""))["ensemble_sentiment"]["score"]
            for p in posts
        ]
        current = scores[-1]
        half = max(1, len(scores) // 2)
        first_avg = sum(scores[:half]) / half
        second_avg = sum(scores[half:]) / max(1, len(scores) - half)
        momentum = second_avg - first_avg
        trend = "IMPROVING" if momentum > 0.05 else "DETERIORATING" if momentum < -0.05 else "STABLE"
        return {
            "current_sentiment": current,
            "average_sentiment": sum(scores) / len(scores),
            "sentiment_trend": trend,
            "momentum": momentum,
            "contrarian_signal": abs(sum(scores) / len(scores)) > 0.8,
        }
