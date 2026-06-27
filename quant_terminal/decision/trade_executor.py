"""Ejecutor de trades con selección de algoritmo de ejecución.

La ejecución real depende de un ``broker_api`` inyectado. En modo simulación
(``broker_api=None``) registra órdenes como paper trading.
"""

from __future__ import annotations

import uuid

import pandas as pd

from .risk_manager import RiskManager


class TradeExecutor:
    def __init__(self, broker_api=None, risk_manager: RiskManager | None = None) -> None:
        self.broker_api = broker_api
        self.risk_manager = risk_manager
        self.execution_algorithms = ["MARKET", "LIMIT", "TWAP", "VWAP", "ICEBERG"]
        self.executed: list[dict] = []

    def _select_execution_algorithm(self, position_size: dict, avg_daily_volume: float) -> str:
        shares = position_size["position_size_shares"]
        ratio = shares / avg_daily_volume if avg_daily_volume else 0
        if ratio > 0.05:
            return "VWAP"
        if ratio > 0.02:
            return "TWAP"
        if ratio > 0.01:
            return "ICEBERG"
        return "MARKET"

    def execute_trade(self, signal: dict, position_size: dict,
                      avg_daily_volume: float = 1e7) -> dict:
        if self.risk_manager is not None:
            risk = self.risk_manager.assess_trade_risk(signal, position_size)
            if not risk["approved"]:
                return {"status": "REJECTED", "reason": risk["recommendations"]}

        algo = self._select_execution_algorithm(position_size, avg_daily_volume)
        order = {
            "trade_id": f"TRD-{uuid.uuid4().hex[:8]}",
            "symbol": signal.get("symbol"),
            "side": signal.get("action", signal.get("final_signal")),
            "quantity": position_size["position_size_shares"],
            "execution_algorithm": algo,
            "stop_loss": signal.get("stop_loss"),
            "take_profit": signal.get("take_profit_1"),
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        if self.broker_api is None:
            order.update({"status": "SIMULATED", "fill_price": signal.get("entry_price")})
        else:
            order.update(self._submit(order))

        self.executed.append(order)
        return order

    def _submit(self, order: dict) -> dict:
        """Hook de integración con un broker real (no implementado aquí)."""
        raise NotImplementedError(
            "Conecta un broker_api concreto (p. ej. Alpaca) para ejecución real."
        )
