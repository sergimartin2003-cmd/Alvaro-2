"""Detección de anomalías (PROMPT 1.9).

Autoencoder (TF opcional), Isolation Forest (sklearn opcional), outliers
estadísticos (numpy: zscore/iqr/mad) y anomalías específicas de mercado por
reglas. Las funciones estadísticas y de mercado funcionan sin libs pesadas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class AnomalyDetectionEngine:
    def train_autoencoder(self, market_data: pd.DataFrame, latent_dim: int = 10,
                          epochs: int = 20) -> dict:
        from tensorflow.keras.layers import Dense, Input
        from tensorflow.keras.models import Model

        X = market_data.fillna(0).values.astype("float32")
        d = X.shape[1]
        inp = Input(shape=(d,))
        e = Dense(max(d // 2, latent_dim), activation="relu")(inp)
        z = Dense(latent_dim, activation="relu")(e)
        dec = Dense(max(d // 2, latent_dim), activation="relu")(z)
        out = Dense(d, activation="linear")(dec)
        model = Model(inp, out)
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, X, epochs=epochs, batch_size=32, verbose=0)
        recon = model.predict(X, verbose=0)
        errors = np.mean((X - recon) ** 2, axis=1)
        threshold = float(np.mean(errors) + 3 * np.std(errors))
        return {
            "model": model,
            "reconstruction_errors": pd.Series(errors, index=market_data.index),
            "threshold": threshold,
            "anomaly_score": pd.Series(errors / (threshold + 1e-9), index=market_data.index),
        }

    def detect_anomalies_isolation_forest(self, features: pd.DataFrame, contamination: float = 0.01) -> pd.Series:
        from sklearn.ensemble import IsolationForest

        X = features.fillna(0).values
        model = IsolationForest(contamination=contamination, random_state=0).fit(X)
        labels = (model.predict(X) == -1).astype(int)
        return pd.Series(labels, index=features.index, name="anomaly")

    def detect_statistical_outliers(self, series: pd.Series, method: str = "zscore",
                                    threshold: float = 3.0) -> pd.Series:
        s = pd.Series(series).astype(float)
        if method == "zscore":
            z = (s - s.mean()) / (s.std() + 1e-9)
            flags = z.abs() > threshold
        elif method == "iqr":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            flags = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)
        elif method == "mad":
            med = s.median()
            mad = (s - med).abs().median()
            flags = (s - med).abs() / (1.4826 * mad + 1e-9) > threshold
        else:
            raise ValueError(f"método desconocido: {method}")
        return flags.astype(int).rename("outlier")

    def detect_market_anomalies(self, market_data: dict) -> list[dict]:
        out = []
        prices = market_data.get("prices")
        volume = market_data.get("volume")
        if prices is not None and len(prices) >= 2:
            prices = pd.Series(prices)
            ret = prices.pct_change()
            recent = ret.tail(5).sum()
            if recent < -0.05:
                out.append({"type": "FLASH_CRASH", "severity": "EXTREME",
                            "market_impact": f"caída {recent:.1%} reciente",
                            "recommended_action": "halt / hedge"})
            gap = ret.iloc[-1] if len(ret) else 0
            if abs(gap) > 0.03:
                out.append({"type": "PRICE_GAP", "severity": "HIGH",
                            "market_impact": f"gap {gap:.1%}",
                            "recommended_action": "verificar noticias"})
        if volume is not None and len(volume) >= 20:
            volume = pd.Series(volume)
            if volume.iloc[-1] > 10 * volume.tail(20).mean():
                out.append({"type": "VOLUME_SPIKE", "severity": "HIGH",
                            "market_impact": "volumen >10x media",
                            "recommended_action": "posible catalizador"})
        vix = market_data.get("vix")
        if vix is not None and vix > 40:
            out.append({"type": "VOLATILITY_SPIKE", "severity": "EXTREME",
                        "market_impact": f"VIX {vix}", "recommended_action": "reducir riesgo"})
        return out
