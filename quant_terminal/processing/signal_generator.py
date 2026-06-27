"""Generación de señales por confluencia de indicadores técnicos."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .technical_indicators import TechnicalIndicatorEngine


class SignalGenerator:
    """Genera señales BUY/SELL/HOLD ponderando categorías de indicadores."""

    def __init__(self, indicator_engine: TechnicalIndicatorEngine | None = None) -> None:
        self.indicator_engine = indicator_engine or TechnicalIndicatorEngine()
        self.signal_weights = {
            "trend": 0.30,
            "momentum": 0.25,
            "volatility": 0.20,
            "volume": 0.15,
            "patterns": 0.10,
        }

    def _ensure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if "rsi" not in df.columns:
            return self.indicator_engine.calculate_all_indicators(df)
        return df

    def generate_signals(self, df: pd.DataFrame) -> dict:
        """Devuelve la señal final con desglose por categoría y niveles de entrada."""
        df = self._ensure_indicators(df)
        last = df.iloc[-1]
        price = float(last["close"])

        categories = {
            "trend": self._trend_signal(last),
            "momentum": self._momentum_signal(last),
            "volatility": self._volatility_signal(last),
            "volume": self._volume_signal(df),
        }

        confluence = self.calculate_confluence_score(categories)
        if confluence > 0.3:
            final = "BUY"
        elif confluence < -0.3:
            final = "SELL"
        else:
            final = "HOLD"

        atr = float(last.get("atr", np.nan))
        levels = self.calculate_entry_exit(df, final, atr) if final != "HOLD" else {}

        return {
            "timestamp": str(df.index[-1]) if df.index.name else None,
            "price": price,
            "signals": categories,
            "final_signal": final,
            "confluence_score": confluence,
            "confidence": min(1.0, abs(confluence)),
            **levels,
        }

    # --- señales por categoría ---
    @staticmethod
    def _sig(strength: float, reasons: list[str]) -> dict:
        if strength > 0.15:
            s = "BUY"
        elif strength < -0.15:
            s = "SELL"
        else:
            s = "HOLD"
        return {"signal": s, "strength": float(abs(strength)), "reasons": reasons}

    def _trend_signal(self, last: pd.Series) -> dict:
        score, reasons = 0.0, []
        if last["close"] > last.get("sma_200", last["close"]):
            score += 0.4
            reasons.append("Precio por encima de SMA 200")
        else:
            score -= 0.4
        if last.get("ema_9", 0) > last.get("ema_21", 0):
            score += 0.3
            reasons.append("EMA 9 > EMA 21")
        else:
            score -= 0.3
        if last.get("macd_diff", 0) > 0:
            score += 0.3
            reasons.append("MACD histogram positivo")
        else:
            score -= 0.3
        return self._sig(np.clip(score, -1, 1), reasons)

    def _momentum_signal(self, last: pd.Series) -> dict:
        score, reasons = 0.0, []
        rsi = last.get("rsi", 50)
        if rsi < 30:
            score += 0.6
            reasons.append(f"RSI sobreventa ({rsi:.0f})")
        elif rsi > 70:
            score -= 0.6
            reasons.append(f"RSI sobrecompra ({rsi:.0f})")
        if last.get("stoch_k", 50) > last.get("stoch_d", 50):
            score += 0.4
            reasons.append("Stochastic cruce alcista")
        else:
            score -= 0.4
        return self._sig(np.clip(score, -1, 1), reasons)

    def _volatility_signal(self, last: pd.Series) -> dict:
        score, reasons = 0.0, []
        if last["close"] < last.get("bb_lower", -np.inf):
            score += 0.8
            reasons.append("Precio bajo banda inferior de Bollinger")
        elif last["close"] > last.get("bb_upper", np.inf):
            score -= 0.8
            reasons.append("Precio sobre banda superior de Bollinger")
        else:
            reasons.append("Precio dentro de bandas")
        return self._sig(np.clip(score, -1, 1), reasons)

    def _volume_signal(self, df: pd.DataFrame) -> dict:
        reasons = []
        obv = df["obv"]
        score = 0.5 if obv.iloc[-1] > obv.iloc[-2] else -0.5
        reasons.append("OBV " + ("subiendo" if score > 0 else "bajando"))
        return self._sig(score, reasons)

    def calculate_confluence_score(self, signals: dict) -> float:
        """Score ponderado entre -1 y +1."""
        total = 0.0
        for category, data in signals.items():
            w = self.signal_weights.get(category, 0.10)
            strength = data["strength"]
            if data["signal"] == "BUY":
                total += w * strength
            elif data["signal"] == "SELL":
                total -= w * strength
        return float(total)

    def calculate_entry_exit(self, df: pd.DataFrame, signal: str, atr: float) -> dict:
        """Calcula entrada, stop y objetivos a partir del ATR."""
        price = float(df["close"].iloc[-1])
        if not np.isfinite(atr) or atr <= 0:
            atr = price * 0.01
        if signal == "BUY":
            stop = price - 1.5 * atr
            tp1, tp2 = price + 2 * atr, price + 3 * atr
        else:  # SELL
            stop = price + 1.5 * atr
            tp1, tp2 = price - 2 * atr, price - 3 * atr
        rr = abs(tp1 - price) / abs(price - stop) if price != stop else 0.0
        return {
            "entry_price": price,
            "stop_loss": round(stop, 2),
            "take_profit_1": round(tp1, 2),
            "take_profit_2": round(tp2, 2),
            "risk_reward_ratio": round(rr, 2),
            "position_size_suggestion": 0.02,
        }
