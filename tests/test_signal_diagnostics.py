"""Unit tests for backtest/signal_diagnostics.py"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.signal_diagnostics import (
    run_filter_funnel,
    run_all_funnels,
    check_signal_frequency,
    save_funnel_csv,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_cfg(
    adx_min: int = 25,
    adx_lookback: int = 5,
    volume_factor: float = 1.5,
) -> dict:
    return {
        "strategy": {
            "ema_fast": 9,
            "ema_mid": 21,
            "ema_slow": 50,
            "rsi_period": 14,
            "rsi_oversold": 45,
            "rsi_overbought": 70,
            "adx_period": 14,
            "adx_min": adx_min,
            "adx_lookback": adx_lookback,
            "atr_period": 14,
            "atr_stop_multiplier": 1.5,
            "volume_factor": volume_factor,
            "spy_ema_trend": 200,
        }
    }


def _make_trending_ohlcv(n: int = 400) -> pd.DataFrame:
    """Simulate a steadily rising price series with clear EMA crossovers."""
    np.random.seed(42)
    idx = pd.bdate_range("2022-01-01", periods=n)
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5 + 0.3)
    vol = np.random.randint(500_000, 5_000_000, n).astype(float)
    return pd.DataFrame({
        "open":   prices * 0.99,
        "high":   prices * 1.01,
        "low":    prices * 0.98,
        "close":  prices,
        "volume": vol,
    }, index=idx)


def _make_spy_close(n: int = 400) -> pd.Series:
    np.random.seed(0)
    idx = pd.bdate_range("2022-01-01", periods=n)
    prices = 400 + np.cumsum(np.random.randn(n) * 0.5 + 0.3)
    return pd.Series(prices, index=idx, name="SPY")


# ── run_filter_funnel ──────────────────────────────────────────────────────────

class TestRunFilterFunnel:
    def test_returns_expected_keys(self):
        df = _make_trending_ohlcv()
        spy = _make_spy_close()
        cfg = _make_cfg()
        result = run_filter_funnel("TEST", df, spy, cfg)
        expected_keys = {
            "symbol", "trading_days", "ema_crossover", "ema_crossover_bull",
            "crossover_volume", "crossover_rsi", "crossover_adx", "final_entries",
        }
        assert set(result.keys()) == expected_keys

    def test_trading_days_equals_df_length(self):
        df = _make_trending_ohlcv(300)
        spy = _make_spy_close(300)
        result = run_filter_funnel("X", df, spy, _make_cfg())
        assert result["trading_days"] == 300

    def test_counts_are_non_negative(self):
        df = _make_trending_ohlcv()
        spy = _make_spy_close()
        result = run_filter_funnel("X", df, spy, _make_cfg())
        for k, v in result.items():
            if k != "symbol":
                assert v >= 0, f"{k} is negative: {v}"

    def test_funnel_is_strictly_non_increasing(self):
        """Each successive filter can only reduce the count, never increase it."""
        df = _make_trending_ohlcv()
        spy = _make_spy_close()
        result = run_filter_funnel("X", df, spy, _make_cfg())
        cols = [
            "ema_crossover", "ema_crossover_bull", "crossover_volume",
            "crossover_rsi", "crossover_adx", "final_entries",
        ]
        counts = [result[c] for c in cols]
        for i in range(1, len(counts)):
            assert counts[i] <= counts[i - 1], (
                f"Step {cols[i]} ({counts[i]}) > step {cols[i-1]} ({counts[i-1]})"
            )

    def test_adx_lookback_relaxes_filter(self):
        """Larger adx_lookback should yield >= signals than lookback=1."""
        df = _make_trending_ohlcv()
        spy = _make_spy_close()
        strict  = run_filter_funnel("X", df, spy, _make_cfg(adx_lookback=1))
        relaxed = run_filter_funnel("X", df, spy, _make_cfg(adx_lookback=10))
        assert relaxed["crossover_adx"] >= strict["crossover_adx"]

    def test_date_slicing_reduces_count(self):
        df = _make_trending_ohlcv(400)
        spy = _make_spy_close(400)
        full   = run_filter_funnel("X", df, spy, _make_cfg())
        sliced = run_filter_funnel("X", df, spy, _make_cfg(),
                                   start="2022-06-01", end="2022-12-31")
        assert sliced["trading_days"] < full["trading_days"]

    def test_zero_adx_min_passes_all_crossovers(self):
        """With adx_min=0, the ADX step should not filter out any crossover."""
        df = _make_trending_ohlcv()
        spy = _make_spy_close()
        result = run_filter_funnel("X", df, spy, _make_cfg(adx_min=0))
        # crossover_rsi should equal crossover_adx since ADX filter can't block
        assert result["crossover_adx"] == result["crossover_rsi"]


# ── run_all_funnels ────────────────────────────────────────────────────────────

class TestRunAllFunnels:
    def test_returns_dataframe_with_symbol_index(self):
        df = _make_trending_ohlcv()
        spy = _make_spy_close()
        ohlcv = {"AAPL": df, "NVDA": df}
        result = run_all_funnels(ohlcv, spy, _make_cfg())
        assert isinstance(result, pd.DataFrame)
        assert set(result.index) == {"AAPL", "NVDA"}

    def test_column_set(self):
        df = _make_trending_ohlcv()
        spy = _make_spy_close()
        result = run_all_funnels({"A": df}, spy, _make_cfg())
        expected = {
            "trading_days", "ema_crossover", "ema_crossover_bull",
            "crossover_volume", "crossover_rsi", "crossover_adx", "final_entries",
        }
        assert expected.issubset(set(result.columns))


# ── check_signal_frequency ────────────────────────────────────────────────────

class TestCheckSignalFrequency:
    def _make_entries(self, n_signals_per_sym: int, n_cols: int = 3) -> pd.DataFrame:
        idx = pd.bdate_range("2022-01-01", periods=500)
        data = np.zeros((500, n_cols), dtype=bool)
        step = max(1, 500 // n_signals_per_sym)
        for row in range(0, 500, step):
            data[row, :] = True
        cols = [f"SYM{i}" for i in range(n_cols)]
        return pd.DataFrame(data, index=idx, columns=cols)

    def test_healthy_frequency_no_warning(self, capsys):
        entries = self._make_entries(n_signals_per_sym=10)
        check_signal_frequency(entries, "2022-01-01", "2024-01-01")
        out = capsys.readouterr().out
        assert "✅" in out

    def test_too_restrictive_warning(self, capsys):
        entries = self._make_entries(n_signals_per_sym=1)  # ~0.5/yr
        check_signal_frequency(entries, "2022-01-01", "2024-01-01")
        out = capsys.readouterr().out
        assert "TOO RESTRICTIVE" in out

    def test_too_noisy_warning(self, capsys):
        entries = self._make_entries(n_signals_per_sym=300)  # ~150/yr
        check_signal_frequency(entries, "2022-01-01", "2024-01-01")
        out = capsys.readouterr().out
        assert "TOO NOISY" in out


# ── save_funnel_csv ────────────────────────────────────────────────────────────

class TestSaveFunnelCsv:
    def test_creates_file(self, tmp_path):
        df = _make_trending_ohlcv()
        spy = _make_spy_close()
        funnel_df = run_all_funnels({"AAPL": df}, spy, _make_cfg())
        path = save_funnel_csv(funnel_df, tmp_path)
        assert path.exists()
        assert path.name == "signal_funnel.csv"

    def test_csv_has_correct_columns(self, tmp_path):
        df = _make_trending_ohlcv()
        spy = _make_spy_close()
        funnel_df = run_all_funnels({"AAPL": df}, spy, _make_cfg())
        path = save_funnel_csv(funnel_df, tmp_path)
        loaded = pd.read_csv(path, index_col="symbol")
        assert "final_entries" in loaded.columns
        assert "trading_days" in loaded.columns
