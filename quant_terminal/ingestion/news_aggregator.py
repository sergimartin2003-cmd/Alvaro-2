"""Agregador de noticias financieras vía RSS (feedparser, perezoso)."""

from __future__ import annotations

from ..processing.sentiment import FinancialSentimentAnalyzer

_DEFAULT_FEEDS = {
    "CNBC": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "MarketWatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "WSJ": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
}

_MARKET_KEYWORDS = [
    "fed", "rate hike", "rate cut", "inflation", "cpi", "gdp", "employment",
    "earnings", "merger", "acquisition", "bankruptcy", "crash", "rally",
    "trump", "tariff", "sanctions", "war", "oil", "gold", "bitcoin",
]


class NewsAggregator:
    def __init__(self, rss_feeds: dict | None = None,
                 sentiment_analyzer: FinancialSentimentAnalyzer | None = None) -> None:
        self.rss_feeds = rss_feeds or dict(_DEFAULT_FEEDS)
        self.sentiment_analyzer = sentiment_analyzer or FinancialSentimentAnalyzer()

    def fetch_all_feeds(self, limit_per_feed: int = 20) -> list[dict]:
        import feedparser

        news = []
        for source, url in self.rss_feeds.items():
            try:
                feed = feedparser.parse(url)
            except Exception:
                continue
            for entry in feed.entries[:limit_per_feed]:
                news.append(
                    {
                        "source": source,
                        "title": getattr(entry, "title", ""),
                        "summary": getattr(entry, "summary", ""),
                        "link": getattr(entry, "link", ""),
                        "published": getattr(entry, "published", ""),
                    }
                )
        return news

    def filter_market_moving(self, news_list: list[dict]) -> list[dict]:
        out = []
        for n in news_list:
            text = f"{n['title']} {n.get('summary','')}".lower()
            if any(k in text for k in _MARKET_KEYWORDS):
                n["market_moving"] = True
                out.append(n)
        return out

    def analyze_news_sentiment(self, news_list: list[dict]) -> dict:
        scored = []
        for n in news_list:
            res = self.sentiment_analyzer.analyze_text(f"{n['title']} {n.get('summary','')}")
            score = res["ensemble_sentiment"]["score"]
            scored.append({**n, "sentiment": res["ensemble_sentiment"], "score": score})
        overall = sum(s["score"] for s in scored) / len(scored) if scored else 0.0
        return {
            "overall_sentiment": overall,
            "articles": scored,
            "market_moving_news": [s for s in scored if abs(s["score"]) > 0.3],
        }
