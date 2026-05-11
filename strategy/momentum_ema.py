from __future__ import annotations

import pandas as pd

from config.settings import (
    MAX_RSI_ENTRY, MAX_RSI_EXIT, MIN_RSI,
    MIN_BARS_REQUIRED, VOLUME_CONFIRMATION_MULTIPLIER, MIN_AVG_DAILY_VOLUME,
    MIN_ADX, ATR_STOP_MULTIPLIER,
)
from strategy.base_strategy import BaseStrategy, EntrySignal, ExitSignal, ScoredCandidate
from strategy.signals import add_indicators, detect_ema_crossover, calculate_volume_ratio


class MomentumEMAStrategy(BaseStrategy):
    """
    Triple-Confirmation EMA Strategy:
    BUY when ALL are true:
      1. EMA-9 crossed above EMA-21 within last 2 bars
      2. Price > EMA-50 (uptrend filter)
      3. Volume > 1.2x 20-day average (institutional confirmation)
      4. RSI between 45-70 (not overbought, not in freefall)

    EXIT (midday override) when ANY is true:
      - EMA-9 crosses below EMA-21
      - Price < EMA-50
      - RSI > 80 (extremely overbought)
    """

    def __init__(self, alpaca_client) -> None:
        self._client = alpaca_client

    def _get_indicators(self, symbol: str) -> pd.DataFrame | None:
        try:
            bars = self._client.get_bars(symbol, "1Day", limit=82)
            # Drop today's incomplete bar during market hours to avoid low partial-day volume
            import pandas as pd
            today = pd.Timestamp.now(tz="UTC").normalize()
            if len(bars) > 0 and bars.index[-1] >= today:
                bars = bars.iloc[:-1]
            if len(bars) < MIN_BARS_REQUIRED:
                return None
            return add_indicators(bars)
        except Exception as exc:
            print(f"[{symbol}] get_bars failed, skipping: {exc}")
            return None

    def check_entry_signal(self, symbol: str) -> EntrySignal | None:
        df = self._get_indicators(symbol)
        if df is None:
            return None

        last = df.iloc[-1]
        reasons = []
        score = 0

        # 1. EMA alignment: EMA-9 must be above EMA-21
        if last["ema_9"] <= last["ema_21"]:
            return None
        crossover = detect_ema_crossover(df["ema_9"], df["ema_21"], lookback=15)
        if crossover == "bullish":
            reasons.append("EMA9 > EMA21 fresh crossover")
            score += 35
        else:
            reasons.append("EMA9 > EMA21 aligned")
            score += 15

        # 2. Price above EMA-50 (trend filter)
        if last["close"] <= last["ema_50"]:
            return None
        reasons.append("Price above EMA50")
        score += 25

        # 3. Volume confirmation
        vol_ratio = calculate_volume_ratio(df["volume"], lookback=20)
        if vol_ratio < VOLUME_CONFIRMATION_MULTIPLIER:
            return None
        reasons.append(f"Volume {vol_ratio:.1f}x avg")
        score += 20

        # 4. RSI filter
        rsi = last["rsi_14"]
        if not (MIN_RSI <= rsi <= MAX_RSI_ENTRY):
            return None
        reasons.append(f"RSI {rsi:.1f}")
        score += 20

        # 5. ADX filter — only trade real trends, not sideways noise
        adx = last["adx_14"]
        if adx < MIN_ADX:
            return None
        reasons.append(f"ADX {adx:.1f}")
        score += 10

        # Minimum liquidity check
        avg_vol = df["volume"].tail(20).mean()
        if avg_vol < MIN_AVG_DAILY_VOLUME:
            return None

        # ATR-based stop: adapts to each stock's actual volatility
        atr = float(last["atr_14"])
        entry = float(last["close"])
        atr_stop = round(entry - ATR_STOP_MULTIPLIER * atr, 2)

        return EntrySignal(
            symbol=symbol,
            entry_price=entry,
            confidence_score=min(score, 100),
            reason=", ".join(reasons),
            atr_stop=atr_stop,
        )

    def check_exit_signal(self, symbol: str, bars: pd.DataFrame | None = None) -> ExitSignal | None:
        if bars is None:
            bars = self._client.get_bars(symbol, "1Day", limit=80)
        if len(bars) < MIN_BARS_REQUIRED:
            return None

        df = add_indicators(bars)
        last = df.iloc[-1]

        # EMA bearish crossover
        crossover = detect_ema_crossover(df["ema_9"], df["ema_21"], lookback=15)
        if crossover == "bearish":
            return ExitSignal(symbol=symbol, reason="EMA9 crossed below EMA21")

        # Price lost trend support
        if last["close"] < last["ema_50"]:
            return ExitSignal(symbol=symbol, reason="Price below EMA50 trend support")

        # Overbought
        if last["rsi_14"] > MAX_RSI_EXIT:
            return ExitSignal(symbol=symbol, reason=f"RSI overbought ({last['rsi_14']:.1f})")

        return None

    def score_candidates(self, symbols: list[str]) -> list[ScoredCandidate]:
        results: list[ScoredCandidate] = []
        for symbol in symbols:
            df = self._get_indicators(symbol)
            if df is None:
                results.append(ScoredCandidate(symbol=symbol, score=0, signals_met=["Insufficient data"]))
                continue

            last = df.iloc[-1]
            signals_met: list[str] = []
            score = 0

            crossover = detect_ema_crossover(df["ema_9"], df["ema_21"], lookback=15)
            if crossover == "bullish":
                signals_met.append("EMA bullish crossover")
                score += 35
            elif last["ema_9"] > last["ema_21"]:
                signals_met.append("EMA9 > EMA21 (no fresh cross)")
                score += 15

            if last["close"] > last["ema_50"]:
                signals_met.append("Above EMA50")
                score += 25

            vol_ratio = calculate_volume_ratio(df["volume"], lookback=20)
            if vol_ratio >= VOLUME_CONFIRMATION_MULTIPLIER:
                signals_met.append(f"Volume {vol_ratio:.1f}x avg")
                score += 20
            elif vol_ratio >= 1.0:
                score += 5

            rsi = last["rsi_14"]
            if MIN_RSI <= rsi <= MAX_RSI_ENTRY:
                signals_met.append(f"RSI {rsi:.1f}")
                score += 20
            elif rsi < MIN_RSI:
                signals_met.append(f"RSI low {rsi:.1f}")

            results.append(ScoredCandidate(symbol=symbol, score=score, signals_met=signals_met))

        return sorted(results, key=lambda c: c.score, reverse=True)
