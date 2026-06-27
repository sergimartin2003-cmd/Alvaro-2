"""Demo end-to-end con datos sintéticos (sin red ni dependencias pesadas).

Genera un OHLCV de mercado simulado, calcula indicadores, genera la señal
técnica, la agrega con señales sintéticas de otras fuentes y la pasa por el
sistema de decisión con gestión de riesgo.

Ejecutar: python examples/run_demo.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_terminal.decision.trading_system import TradingSystem
from quant_terminal.processing.stochastic_models import OrnsteinUhlenbeckModel


def synthetic_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Random walk con leve drift alcista + clustering de volatilidad.
    rets = rng.normal(0.0005, 0.012, n)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = rng.integers(1_000_000, 5_000_000, n)
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )


def main() -> None:
    df = synthetic_ohlcv()
    print(f"Datos sintéticos: {len(df)} barras, último cierre = {df['close'].iloc[-1]:.2f}\n")

    # Modelo estocástico de mean reversion sobre el precio.
    ou = OrnsteinUhlenbeckModel(df["close"].values)
    params = ou.fit()
    print(f"Ornstein-Uhlenbeck: half-life={params['half_life']:.1f} días, mu={params['mu']:.2f}")
    print("Entrada óptima:", ou.predict_optimal_entry(df["close"].iloc[-1]), "\n")

    system = TradingSystem({"portfolio_value": 100_000})
    extra = {
        "sentiment_analysis": {"signal": "BUY", "confidence": 0.6, "strength": 0.6},
        "macro_analysis": {"signal": "HOLD", "confidence": 0.5, "strength": 0.4},
        "options_flow": {"signal": "BUY", "confidence": 0.7, "strength": 0.65},
    }
    result = system.process_symbol("SPY", df, extra_signals=extra)

    decision = result["decision"]
    print("=== DECISIÓN ===")
    print(f"  Acción: {decision['action']}  (confianza {decision['confidence']:.2f})")
    print(f"  Score agregado: {decision['aggregate_score']:.3f}")
    print(f"  P(mercado sube) bayesiano: {decision['bayesian_probability']:.2%}")
    if "position_size" in result:
        ps = result["position_size"]
        print(f"  Tamaño: {ps['position_size_shares']} acciones (${ps['position_size_dollars']:.0f})")
        print(f"  Riesgo aprobado: {result['risk_assessment']['approved']} "
              f"(nivel {result['risk_assessment']['risk_level']})")
    if "trade_result" in result:
        print(f"  Ejecución: {result['trade_result']['status']} "
              f"via {result['trade_result']['execution_algorithm']}")


if __name__ == "__main__":
    main()
