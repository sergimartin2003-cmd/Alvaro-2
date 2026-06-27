import pandas as pd

from quant_terminal.decision.risk_manager import RiskManager
from quant_terminal.decision.trading_system import TradingSystem
from quant_terminal.processing.risk_parity import RiskParityOptimizer


def test_position_sizing_respects_risk():
    rm = RiskManager(portfolio_value=100_000, max_risk_per_trade=0.02)
    pos = rm.calculate_position_size({"risk_reward_ratio": 2.5}, asset_price=150, stop_loss_price=147)
    assert pos["risk_amount"] == 2000  # 2% de 100k
    assert pos["position_size_shares"] > 0
    assert pos["approved"] is True


def test_var_positive():
    rm = RiskManager(portfolio_value=100_000)
    var = rm.calculate_portfolio_var()
    assert var["var_95"] > 0
    assert var["var_99"] >= var["var_95"]


def test_risk_parity_weights_sum_to_one():
    import numpy as np

    rng = np.random.default_rng(3)
    rets = pd.DataFrame(rng.normal(0, 0.01, (250, 4)), columns=["A", "B", "C", "D"])
    res = RiskParityOptimizer(rets).optimize()
    assert abs(sum(res["weights"].values()) - 1.0) < 1e-3
    assert res["portfolio_volatility"] > 0


def test_trading_system_end_to_end(ohlcv):
    system = TradingSystem({"portfolio_value": 100_000})
    result = system.process_symbol("SPY", ohlcv)
    assert result["symbol"] == "SPY"
    assert result["decision"]["action"] in ("BUY", "SELL", "HOLD")
