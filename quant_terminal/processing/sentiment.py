"""Análisis de sentimiento financiero (FinBERT + VADER) y extracción de entidades.

FinBERT/VADER/spaCy se cargan de forma perezosa. Si no están instalados, el
analizador degrada a un léxico básico para que el resto del sistema funcione.
"""

from __future__ import annotations

import re

_POSITIVE = {"beat", "rally", "surge", "gain", "bullish", "upgrade", "growth", "strong", "record"}
_NEGATIVE = {"miss", "crash", "plunge", "loss", "bearish", "downgrade", "recession", "weak", "cut", "hike"}

_TICKER_RE = re.compile(r"\$[A-Z]{1,5}\b|\b[A-Z]{2,5}\b")


class FinancialSentimentAnalyzer:
    """Sentimiento por ensemble FinBERT+VADER con fallback léxico."""

    def __init__(self) -> None:
        self.finbert = None
        self.vader = None
        self._init_models()

    def _init_models(self) -> None:
        try:
            from transformers import pipeline

            self.finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        except Exception:
            self.finbert = None
        try:
            import nltk
            from nltk.sentiment import SentimentIntensityAnalyzer

            try:
                self.vader = SentimentIntensityAnalyzer()
            except LookupError:
                nltk.download("vader_lexicon", quiet=True)
                self.vader = SentimentIntensityAnalyzer()
        except Exception:
            self.vader = None

    def _lexicon_score(self, text: str) -> float:
        words = set(re.findall(r"[a-z]+", text.lower()))
        pos = len(words & _POSITIVE)
        neg = len(words & _NEGATIVE)
        if pos == neg == 0:
            return 0.0
        return (pos - neg) / (pos + neg)

    def analyze_text(self, text: str, source: str = "news") -> dict:
        finbert_res = None
        if self.finbert is not None:
            try:
                r = self.finbert(text[:512])[0]
                finbert_res = {"label": r["label"].lower(), "confidence": float(r["score"])}
            except Exception:
                finbert_res = None

        vader_res = None
        if self.vader is not None:
            try:
                s = self.vader.polarity_scores(text)
                vader_res = {
                    "compound": s["compound"],
                    "positive": s["pos"],
                    "negative": s["neg"],
                    "neutral": s["neu"],
                }
            except Exception:
                vader_res = None

        # Ensemble: combina lo disponible con el léxico de respaldo.
        scores = [self._lexicon_score(text)]
        if finbert_res:
            mapping = {"positive": 1, "negative": -1, "neutral": 0}
            scores.append(mapping.get(finbert_res["label"], 0) * finbert_res["confidence"])
        if vader_res:
            scores.append(vader_res["compound"])
        ens_score = sum(scores) / len(scores)
        label = "positive" if ens_score > 0.1 else "negative" if ens_score < -0.1 else "neutral"

        return {
            "text": text,
            "finbert_sentiment": finbert_res,
            "vader_sentiment": vader_res,
            "ensemble_sentiment": {
                "label": label,
                "score": ens_score,
                "confidence": min(1.0, abs(ens_score) + 0.5),
            },
        }

    def analyze_batch(self, texts: list[str], source: str = "news") -> list[dict]:
        return [self.analyze_text(t, source) for t in texts]

    def extract_financial_entities(self, text: str) -> dict:
        tickers = sorted(set(m.lstrip("$") for m in _TICKER_RE.findall(text)))
        return {"tickers": tickers}

    def detect_sentiment_shift(self, historical_sentiments: list[float], window: int = 10) -> dict:
        if len(historical_sentiments) < window + 1:
            return {"shift_detected": False}
        current = historical_sentiments[-1]
        avg = sum(historical_sentiments[-window - 1 : -1]) / window
        magnitude = current - avg
        return {
            "shift_detected": abs(magnitude) > 0.3,
            "shift_direction": "POSITIVE" if magnitude > 0 else "NEGATIVE",
            "shift_magnitude": magnitude,
            "current_sentiment": current,
            "historical_average": avg,
            "significance": "HIGH" if abs(magnitude) > 0.5 else "MEDIUM" if abs(magnitude) > 0.3 else "LOW",
        }
