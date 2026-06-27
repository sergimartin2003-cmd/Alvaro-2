"""Streaming y análisis de cuentas de Twitter/X (tweepy, perezoso)."""

from __future__ import annotations

from ..processing.social_media import SocialMediaAnalyzer


class TwitterStreamAnalyzer:
    def __init__(self, api_key: str, api_secret: str,
                 access_token: str | None = None, access_secret: str | None = None) -> None:
        self.creds = {
            "api_key": api_key,
            "api_secret": api_secret,
            "access_token": access_token,
            "access_secret": access_secret,
        }
        self._client = None
        self.analyzer = SocialMediaAnalyzer()

    @property
    def client(self):
        if self._client is None:
            import tweepy

            self._client = tweepy.Client(
                consumer_key=self.creds["api_key"],
                consumer_secret=self.creds["api_secret"],
                access_token=self.creds["access_token"],
                access_token_secret=self.creds["access_secret"],
            )
        return self._client

    def analyze_tweet_impact(self, tweet: dict) -> dict:
        """Analiza el impacto potencial de un tweet ya recuperado."""
        return self.analyzer.analyze_political_post(tweet)

    async def stream_political_accounts(self, account_ids: list[str], on_post=None):
        """Sondea las cuentas indicadas y procesa cada tweet nuevo.

        ``on_post`` es un callback async opcional para reenviar a Kafka.
        """
        import asyncio

        import tweepy

        seen: set[str] = set()
        while True:
            for account_id in account_ids:
                try:
                    resp = self.client.get_users_tweets(id=account_id, max_results=5)
                except tweepy.TweepyException:
                    continue
                for t in resp.data or []:
                    if t.id in seen:
                        continue
                    seen.add(t.id)
                    post = {"author_id": account_id, "text": t.text}
                    analysis = self.analyze_tweet_impact(post)
                    if on_post is not None:
                        await on_post({**post, "analysis": analysis})
            await asyncio.sleep(30)
