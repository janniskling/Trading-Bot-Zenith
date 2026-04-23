"""Pure technical indicator functions — no external dependencies beyond pandas/numpy."""
from __future__ import annotations

import pandas as pd
import numpy as np


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def detect_ema_crossover(fast: pd.Series, slow: pd.Series, lookback: int = 2) -> str | None:
    """Returns 'bullish', 'bearish', or None depending on recent crossover."""
    if len(fast) < lookback + 1 or len(slow) < lookback + 1:
        return None
    recent_fast = fast.iloc[-(lookback + 1):]
    recent_slow = slow.iloc[-(lookback + 1):]
    was_below = (recent_fast.iloc[0] < recent_slow.iloc[0])
    now_above = (recent_fast.iloc[-1] > recent_slow.iloc[-1])
    if was_below and now_above:
        return "bullish"
    was_above = (recent_fast.iloc[0] > recent_slow.iloc[0])
    now_below = (recent_fast.iloc[-1] < recent_slow.iloc[-1])
    if was_above and now_below:
        return "bearish"
    return None


def calculate_momentum(series: pd.Series, period: int = 20) -> float:
    """Returns % price change over the last `period` bars."""
    if len(series) < period + 1:
        return 0.0
    return float((series.iloc[-1] / series.iloc[-(period + 1)] - 1) * 100)


def calculate_volume_ratio(volume: pd.Series, lookback: int = 20) -> float:
    """Current volume vs. rolling average. >1.2 means above-average volume."""
    if len(volume) < lookback:
        return 0.0
    avg = volume.iloc[-lookback:-1].mean()
    if avg == 0:
        return 0.0
    return float(volume.iloc[-1] / avg)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds EMA, RSI, and volume columns in place."""
    df = df.copy()
    df["ema_9"] = calculate_ema(df["close"], 9)
    df["ema_21"] = calculate_ema(df["close"], 21)
    df["ema_50"] = calculate_ema(df["close"], 50)
    df["rsi_14"] = calculate_rsi(df["close"], 14)
    df["vol_sma_20"] = df["volume"].rolling(20).mean()
    return df
