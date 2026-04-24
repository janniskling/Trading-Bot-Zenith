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


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(com=period - 1, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(com=period - 1, min_periods=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(com=period - 1, min_periods=period).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(com=period - 1, min_periods=period).mean()


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
    df = df.copy()
    df["ema_9"] = calculate_ema(df["close"], 9)
    df["ema_21"] = calculate_ema(df["close"], 21)
    df["ema_50"] = calculate_ema(df["close"], 50)
    df["rsi_14"] = calculate_rsi(df["close"], 14)
    df["vol_sma_20"] = df["volume"].rolling(20).mean()
    df["atr_14"] = calculate_atr(df, 14)
    df["adx_14"] = calculate_adx(df, 14)
    return df
