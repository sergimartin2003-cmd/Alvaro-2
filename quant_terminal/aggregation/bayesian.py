"""Combinación bayesiana de señales (actualización de creencia direccional)."""

from __future__ import annotations


class BayesianDecisionEngine:
    """Actualiza P(mercado sube) con el teorema de Bayes a partir de señales."""

    def __init__(self) -> None:
        self.prior_probabilities = {"market_up": 0.50, "market_down": 0.50}
        self.source_reliability = {
            "technical_analysis": 0.65,
            "sentiment_analysis": 0.58,
            "macro_analysis": 0.70,
            "options_flow": 0.72,
            "ml_models": 0.63,
            "seasonality": 0.60,
        }

    def bayesian_update(self, signals_dict: dict) -> dict:
        prob_up = self.prior_probabilities["market_up"]
        prob_down = 1 - prob_up
        summary = []

        for source, data in signals_dict.items():
            if source not in self.source_reliability:
                continue
            signal = data.get("signal", "HOLD")
            if signal == "HOLD":
                continue
            conf = data.get("confidence", 0.5)
            reliability = self.source_reliability[source]
            adj = reliability * conf + 0.5 * (1 - conf)

            if signal in ("BUY", "STRONG_BUY"):
                p_if_up, p_if_down = adj, 1 - adj
            else:  # SELL
                p_if_up, p_if_down = 1 - adj, adj

            denom = p_if_up * prob_up + p_if_down * prob_down
            prob_up = (p_if_up * prob_up) / denom if denom else prob_up
            prob_down = 1 - prob_up
            summary.append(
                {"source": source, "signal": signal, "likelihood": adj, "posterior": prob_up}
            )

        if prob_up > 0.75:
            action = "STRONG_BUY"
        elif prob_up > 0.60:
            action = "BUY"
        elif prob_up < 0.25:
            action = "STRONG_SELL"
        elif prob_up < 0.40:
            action = "SELL"
        else:
            action = "HOLD"

        return {
            "posterior_probabilities": {"market_up": prob_up, "market_down": prob_down},
            "evidence_strength": max(prob_up, prob_down),
            "update_summary": summary,
            "decision": {
                "action": action,
                "confidence": max(prob_up, prob_down),
                "reasoning": f"Probabilidad posterior tras Bayes: {max(prob_up, prob_down):.2%}",
            },
        }

    def update_source_reliability(self, source: str, was_correct: bool, alpha: float = 0.1) -> None:
        if source in self.source_reliability:
            current = self.source_reliability[source]
            target = 1.0 if was_correct else 0.0
            self.source_reliability[source] = alpha * target + (1 - alpha) * current
