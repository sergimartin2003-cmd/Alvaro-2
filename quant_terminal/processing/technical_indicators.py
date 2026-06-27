"""Motor de indicadores técnicos.

Calcula trend / momentum / volatility / volume sobre un DataFrame OHLCV usando
la librería ``ta``. Vectorizado con pandas/numpy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:  # ``ta`` es parte del núcleo, pero degradamos con un mensaje claro.
    import ta
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "El paquete 'ta' es necesario para los indicadores técnicos. "
        "Instala con: pip install ta"
    ) from exc


_REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


class TechnicalIndicatorEngine:
    """Calcula el conjunto completo de indicadores técnicos."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas OHLCV: {missing}")

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Devuelve una copia del DataFrame con todas las columnas de indicadores."""
        self._validate(df)
        df = df.copy()
        c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

        # --- Trend ---
        for w in (9, 21, 50, 100, 200):
            df[f"ema_{w}"] = ta.trend.ema_indicator(c, window=w)
        for w in (10, 20, 50, 100, 200):
            df[f"sma_{w}"] = ta.trend.sma_indicator(c, window=w)
        df["macd"] = ta.trend.macd(c)
        df["macd_signal"] = ta.trend.macd_signal(c)
        df["macd_diff"] = ta.trend.macd_diff(c)
        df["adx"] = ta.trend.adx(h, l, c)
        ich = ta.trend.IchimokuIndicator(h, l)
        df["ichimoku_a"] = ich.ichimoku_a()
        df["ichimoku_b"] = ich.ichimoku_b()
        df["psar"] = ta.trend.PSARIndicator(h, l, c).psar()

        # --- Momentum ---
        df["rsi"] = ta.momentum.rsi(c, window=14)
        df["stoch_k"] = ta.momentum.stoch(h, l, c)
        df["stoch_d"] = ta.momentum.stoch_signal(h, l, c)
        df["cci"] = ta.trend.cci(h, l, c, window=20)
        df["williams_r"] = ta.momentum.williams_r(h, l, c)
        df["roc"] = ta.momentum.roc(c, window=10)

        # --- Volatility ---
        bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["atr"] = ta.volatility.average_true_range(h, l, c, window=14)
        kc = ta.volatility.KeltnerChannel(h, l, c, window=20)
        df["kc_upper"] = kc.keltner_channel_hband()
        df["kc_lower"] = kc.keltner_channel_lband()
        log_ret = np.log(c / c.shift(1))
        for w in (10, 20, 30):
            df[f"hist_vol_{w}"] = log_ret.rolling(w).std() * np.sqrt(252)

        # --- Volume ---
        df["obv"] = ta.volume.on_balance_volume(c, v)
        df["vwap"] = self._vwap(df)
        df["mfi"] = ta.volume.money_flow_index(h, l, c, v)
        df["adl"] = ta.volume.acc_dist_index(h, l, c, v)
        df["cmf"] = ta.volume.chaikin_money_flow(h, l, c, v)

        return df

    @staticmethod
    def _vwap(df: pd.DataFrame) -> pd.Series:
        typical = (df["high"] + df["low"] + df["close"]) / 3
        cum_vol = df["volume"].cumsum()
        cum_tp_vol = (typical * df["volume"]).cumsum()
        return cum_tp_vol / cum_vol.replace(0, np.nan)

    def identify_support_resistance(
        self, df: pd.DataFrame, lookback: int = 100
    ) -> dict:
        """Identifica soportes/resistencias por extremos locales (fractales)."""
        self._validate(df)
        window = df.tail(lookback)
        highs, lows = window["high"].values, window["low"].values
        n = len(window)
        resistance, support = [], []
        for i in range(2, n - 2):
            if highs[i] == max(highs[i - 2 : i + 3]):
                resistance.append(float(highs[i]))
            if lows[i] == min(lows[i - 2 : i + 3]):
                support.append(float(lows[i]))

        price = float(window["close"].iloc[-1])
        res = sorted(set(round(r, 2) for r in resistance if r > price))
        sup = sorted(set(round(s, 2) for s in support if s < price), reverse=True)
        return {
            "support_levels": sup[:5],
            "resistance_levels": res[:5],
            "current_price": price,
            "nearest_support": sup[0] if sup else None,
            "nearest_resistance": res[0] if res else None,
        }
