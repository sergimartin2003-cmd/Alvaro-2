"""Trading con Reinforcement Learning (PROMPT 2.2).

Incluye un entorno de trading mínimo (duck-typed estilo Gym) implementado en
numpy puro y testeable, más wrappers para entrenar/backtestear con
stable-baselines3 (import perezoso).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class TradingEnvironment:
    """Entorno de trading simple: posiciones {-1,0,1}, reward = PnL - costes.

    Compatible con la API de Gym (reset/step) sin depender de gym.
    """

    ACTIONS = {0: 0, 1: 1, 2: -1}  # HOLD, LONG, SHORT (posición objetivo)

    def __init__(self, prices: pd.Series, transaction_cost: float = 0.0005,
                 window: int = 10) -> None:
        self.prices = np.asarray(prices, dtype=float)
        self.returns = np.diff(self.prices) / self.prices[:-1]
        self.transaction_cost = transaction_cost
        self.window = window
        self.reset()

    def reset(self):
        self.t = self.window
        self.position = 0
        self.cash = 1.0
        self.equity = 1.0
        self.done = False
        return self._state()

    def _state(self) -> np.ndarray:
        recent = self.returns[self.t - self.window : self.t]
        return np.concatenate([recent, [self.position]]).astype(float)

    def step(self, action: int):
        target = self.ACTIONS.get(int(action), 0)
        cost = self.transaction_cost * abs(target - self.position)
        self.position = target
        r = self.returns[self.t] if self.t < len(self.returns) else 0.0
        reward = self.position * r - cost
        self.equity *= 1 + reward
        self.t += 1
        self.done = self.t >= len(self.returns)
        return self._state(), float(reward), self.done, {"equity": self.equity}

    @property
    def action_space_n(self) -> int:
        return len(self.ACTIONS)


class ReinforcementLearningTrader:
    def build_trading_environment(self, prices, **kwargs) -> TradingEnvironment:
        return TradingEnvironment(prices, **kwargs)

    def train_ppo_agent(self, env, total_timesteps: int = 100000) -> dict:
        from stable_baselines3 import PPO  # type: ignore

        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=total_timesteps)
        return {"trained_model": model}

    def backtest_rl_strategy(self, policy_fn, prices, **kwargs) -> dict:
        """Backtest genérico: policy_fn(state)->action. Funciona con cualquier
        política (RL entrenada o heurística), sin libs externas."""
        env = self.build_trading_environment(prices, **kwargs)
        state = env.reset()
        equity_curve = [env.equity]
        rewards = []
        while not env.done:
            action = policy_fn(state)
            state, reward, done, info = env.step(action)
            equity_curve.append(info["equity"])
            rewards.append(reward)
        rewards = np.array(rewards)
        sharpe = float(rewards.mean() / (rewards.std() + 1e-9) * np.sqrt(252)) if len(rewards) else 0.0
        eq = np.array(equity_curve)
        dd = float((eq / np.maximum.accumulate(eq) - 1).min()) if len(eq) else 0.0
        return {
            "equity_curve": pd.Series(equity_curve),
            "final_equity": float(equity_curve[-1]),
            "sharpe_ratio": sharpe,
            "max_drawdown": dd,
            "n_steps": len(rewards),
        }
