"""Provisión de liquidez / market making (PROMPT 3.3).

Cotizaciones óptimas Avellaneda-Stoikov, gestión de inventario y simulación de
una estrategia de market making. numpy/pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


class LiquidityProvider:
    def calculate_optimal_quotes(self, mid_price: float, inventory: float = 0.0,
                                 risk_aversion: float = 1.0, sigma: float = 0.02,
                                 T: float = 1.0, t: float = 0.0, max_inventory: float = 100.0,
                                 order_arrival: float = 1.5) -> dict:
        tau = max(T - t, 1e-6)
        gamma = risk_aversion
        q = inventory / max(max_inventory, 1)
        reservation = mid_price - q * gamma * sigma**2 * tau
        spread = gamma * sigma**2 * tau + (2 / gamma) * np.log(1 + gamma / order_arrival)
        return {
            "reservation_price": float(reservation),
            "optimal_spread": float(spread),
            "bid_price": float(reservation - spread / 2),
            "ask_price": float(reservation + spread / 2),
            "expected_profit_per_trade": float(spread / 2),
        }

    def manage_inventory(self, current_inventory: float, max_inventory: float,
                         target_inventory: float = 0.0) -> dict:
        deviation = current_inventory - target_inventory
        utilization = abs(current_inventory) / max(max_inventory, 1)
        # Sesgar cotizaciones para volver al objetivo.
        skew = -deviation / max(max_inventory, 1)
        actions = []
        if utilization > 0.8:
            actions.append("REDUCE_POSITION")
        if utilization > 0.95:
            actions.append("STOP_QUOTING_SAME_SIDE")
        return {
            "quote_adjustment": {"skew": float(skew)},
            "inventory_risk": float(utilization),
            "recommended_actions": actions or ["NORMAL"],
        }

    def simulate_market_making_strategy(self, prices: pd.Series, sigma: float = 0.02,
                                        risk_aversion: float = 0.5, fill_prob: float = 0.5,
                                        max_inventory: float = 100.0, seed: int | None = None) -> dict:
        rng = np.random.default_rng(seed)
        p = np.asarray(prices, dtype=float)
        n = len(p)
        inventory, cash = 0.0, 0.0
        equity_curve, inv_curve, trades = [], [], 0
        for i in range(n):
            q = self.calculate_optimal_quotes(p[i], inventory, risk_aversion, sigma,
                                              T=1.0, t=i / n, max_inventory=max_inventory)
            # ¿Se ejecutan bid/ask?
            if rng.random() < fill_prob and inventory < max_inventory:
                inventory += 1
                cash -= q["bid_price"]
                trades += 1
            if rng.random() < fill_prob and inventory > -max_inventory:
                inventory -= 1
                cash += q["ask_price"]
                trades += 1
            equity_curve.append(cash + inventory * p[i])
            inv_curve.append(inventory)
        eq = np.array(equity_curve)
        rets = np.diff(eq)
        sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252)) if len(rets) else 0.0
        dd = float((eq - np.maximum.accumulate(eq)).min()) if len(eq) else 0.0
        return {
            "equity_curve": pd.Series(equity_curve),
            "inventory_evolution": pd.Series(inv_curve),
            "n_trades": trades,
            "final_pnl": float(eq[-1]) if len(eq) else 0.0,
            "sharpe_ratio": sharpe,
            "max_drawdown": dd,
        }
