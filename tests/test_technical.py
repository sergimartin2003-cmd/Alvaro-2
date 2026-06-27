from quant_terminal.processing.signal_generator import SignalGenerator
from quant_terminal.processing.technical_indicators import TechnicalIndicatorEngine


def test_indicators_present(ohlcv):
    df = TechnicalIndicatorEngine().calculate_all_indicators(ohlcv)
    for col in ("rsi", "macd", "bb_upper", "atr", "obv", "vwap", "ema_200"):
        assert col in df.columns
    assert df["rsi"].dropna().between(0, 100).all()


def test_support_resistance(ohlcv):
    sr = TechnicalIndicatorEngine().identify_support_resistance(ohlcv)
    assert sr["current_price"] > 0
    assert isinstance(sr["support_levels"], list)


def test_signal_generation(ohlcv):
    sig = SignalGenerator().generate_signals(ohlcv)
    assert sig["final_signal"] in ("BUY", "SELL", "HOLD")
    assert 0.0 <= sig["confidence"] <= 1.0
    assert set(sig["signals"]) == {"trend", "momentum", "volatility", "volume"}


def test_missing_columns_raises():
    import pandas as pd
    import pytest

    with pytest.raises(ValueError):
        TechnicalIndicatorEngine().calculate_all_indicators(pd.DataFrame({"close": [1, 2, 3]}))
