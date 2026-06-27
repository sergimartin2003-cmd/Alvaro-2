"""Sistema integrado: orquesta análisis, agregación, riesgo y ejecución."""

from __future__ import annotations

import pandas as pd

from ..aggregation.bayesian import BayesianDecisionEngine
from ..aggregation.signal_aggregator import SignalAggregator
from ..processing.signal_generator import SignalGenerator
from ..processing.technical_indicators import TechnicalIndicatorEngine
from .risk_manager import RiskManager
from .trade_executor import TradeExecutor


class TradingSystem:
    """Pipeline de decisión end-to-end (sin dependencias de red por defecto)."""

    def __init__(self, config: dict | None = None) -> None:
        config = config or {}
        self.technical_engine = TechnicalIndicatorEngine()
        self.signal_generator = SignalGenerator(self.technical_engine)
        self.signal_aggregator = SignalAggregator()
        self.bayesian_engine = BayesianDecisionEngine()
        self.risk_manager = RiskManager(
            config.get("portfolio_value", 100000),
            config.get("max_risk_per_trade", 0.02),
            config.get("max_total_exposure", 0.20),
        )
        self.trade_executor = TradeExecutor(
            broker_api=config.get("broker_api"), risk_manager=self.risk_manager
        )

    def process_symbol(self, symbol: str, ohlcv: pd.DataFrame,
                       extra_signals: dict | None = None) -> dict:
        """Procesa un símbolo y devuelve la decisión (y ejecución si procede)."""
        technical = self.signal_generator.generate_signals(ohlcv)

        all_signals = {
            "technical_analysis": {
                "signal": technical["final_signal"],
                "confidence": technical["confidence"],
                "strength": abs(technical["confluence_score"]),
            }
        }
        if extra_signals:
            all_signals.update(extra_signals)

        aggregate = self.signal_aggregator.aggregate_signals(all_signals)
        bayesian = self.bayesian_engine.bayesian_update(all_signals)
        final = self._make_final_decision(symbol, technical, aggregate, bayesian)

        if final["action"] in ("BUY", "SELL"):
            pos = self.risk_manager.calculate_position_size(
                final, technical["price"], technical.get("stop_loss", technical["price"] * 0.98)
            )
            risk = self.risk_manager.assess_trade_risk(final, pos)
            result = {
                "symbol": symbol,
                "decision": final,
                "position_size": pos,
                "risk_assessment": risk,
            }
            if risk["approved"]:
                result["trade_result"] = self.trade_executor.execute_trade(final, pos)
            return result

        return {"symbol": symbol, "decision": final, "action_taken": "NONE"}

    def _make_final_decision(self, symbol, technical, aggregate, bayesian) -> dict:
        agg_score = aggregate["aggregate_score"]
        bayes_prob = bayesian["posterior_probabilities"]["market_up"]
        final_score = 0.6 * agg_score + 0.4 * (bayes_prob - 0.5) * 2
        if final_score > 0.3:
            action = "BUY"
        elif final_score < -0.3:
            action = "SELL"
        else:
            action = "HOLD"
        return {
            "symbol": symbol,
            "action": action,
            "confidence": abs(final_score),
            "aggregate_score": agg_score,
            "bayesian_probability": bayes_prob,
            "entry_price": technical["price"],
            "stop_loss": technical.get("stop_loss"),
            "take_profit_1": technical.get("take_profit_1"),
            "risk_reward_ratio": technical.get("risk_reward_ratio", 0.0),
            "reasoning": f"Score combinado: {final_score:.2f}",
        }
