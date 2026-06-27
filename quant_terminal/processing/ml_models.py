"""Modelos de ML para trading: híbrido LSTM+XGBoost y transformer de series.

Las dependencias (tensorflow, xgboost, scikit-learn, transformers) se importan
de forma perezosa dentro de los métodos para mantener ligero el núcleo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class HybridLSTMXGBoost:
    """LSTM (secuencial) + XGBoost (features) para forecasting de precio."""

    def __init__(self, lookback: int = 60) -> None:
        self.lookback = lookback
        self.lstm_model = None
        self.xgb_model = None
        self.scaler = None
        self.features = ["rsi", "macd", "bb_upper", "bb_lower", "atr", "obv", "vwap"]

    def prepare_data(self, df: pd.DataFrame):
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        cols = [f for f in self.features if f in df.columns] + ["close"]
        scaled = self.scaler.fit_transform(df[cols].dropna())
        X_lstm, y = [], []
        for i in range(self.lookback, len(scaled)):
            X_lstm.append(scaled[i - self.lookback : i, :])
            y.append(scaled[i, -1])
        X_xgb = scaled[self.lookback :, :]
        return np.array(X_lstm), np.array(y), X_xgb

    def build_lstm_model(self, input_shape):
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.models import Sequential

        model = Sequential(
            [
                LSTM(50, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        return model

    def fit(self, X_lstm, y, X_xgb, epochs: int = 50, batch_size: int = 32):
        from xgboost import XGBRegressor

        self.lstm_model = self.build_lstm_model((X_lstm.shape[1], X_lstm.shape[2]))
        self.lstm_model.fit(
            X_lstm, y, epochs=epochs, batch_size=batch_size,
            validation_split=0.1, verbose=0,
        )
        lstm_pred = self.lstm_model.predict(X_lstm, verbose=0)
        X_xgb_enh = np.column_stack([X_xgb, lstm_pred])
        self.xgb_model = XGBRegressor(n_estimators=100, max_depth=5)
        self.xgb_model.fit(X_xgb_enh, y)
        return self

    def predict(self, X_lstm_recent, X_xgb_recent):
        lstm_pred = self.lstm_model.predict(X_lstm_recent, verbose=0)
        X_xgb_enh = np.column_stack([X_xgb_recent, lstm_pred])
        return self.xgb_model.predict(X_xgb_enh)

    @staticmethod
    def generate_signal(current_price: float, predicted_price: float, threshold: float = 0.01):
        exp_ret = (predicted_price - current_price) / current_price
        if exp_ret > threshold:
            return "BUY", exp_ret
        if exp_ret < -threshold:
            return "SELL", exp_ret
        return "HOLD", exp_ret


class TimeSeriesTransformer:
    """Wrapper para transformers de series temporales (Autoformer/Informer)."""

    def __init__(self, seq_len: int = 96, label_len: int = 48, pred_len: int = 24) -> None:
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.model = None

    def build_model(self):
        from transformers import AutoformerConfig, AutoformerForPrediction

        config = AutoformerConfig(
            prediction_length=self.pred_len,
            context_length=self.seq_len,
            num_attention_heads=8,
            dropout=0.1,
        )
        self.model = AutoformerForPrediction(config)
        return self.model
