"""Unit tests for backtest/correlation_filter.py"""
import numpy as np
import pandas as pd

from backtest.correlation_filter import (
    _build_symbol_to_sector,
    check_correlation_constraint,
    apply_correlation_filter,
)


def _make_returns(n: int = 80) -> pd.DataFrame:
    """Synthetic returns: AAPL and MSFT highly correlated, TSLA uncorrelated."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2023-01-01", periods=n)
    base = rng.standard_normal(n)
    aapl = base + rng.standard_normal(n) * 0.05
    msft = base + rng.standard_normal(n) * 0.05  # ~corr 0.95 with AAPL
    tsla = rng.standard_normal(n) * 0.02          # independent
    return pd.DataFrame({"AAPL": aapl, "MSFT": msft, "TSLA": tsla}, index=dates)


def _make_prices(n: int = 80) -> pd.DataFrame:
    returns = _make_returns(n)
    prices = (1 + returns).cumprod() * 100
    return prices


def _make_cfg(mode: str = "reduce", threshold: float = 0.70) -> dict:
    return {
        "correlation": {
            "threshold": threshold,
            "lookback_days": 60,
            "mode": mode,
            "size_reduction": 0.5,
        },
        "sector_caps": {
            "max_per_sector": 3,
            "max_etf": 2,
            "sectors": {
                "Tech": ["AAPL", "MSFT"],
                "Other": ["TSLA"],
            },
        },
    }


# ── check_correlation_constraint ──────────────────────────────────────────────

class TestCheckCorrelationConstraint:
    def setup_method(self):
        self.returns = _make_returns()

    def test_no_open_positions_always_allowed(self):
        allowed, corrs = check_correlation_constraint("AAPL", [], self.returns)
        assert allowed is True
        assert corrs == {}

    def test_uncorrelated_symbol_allowed(self):
        allowed, corrs = check_correlation_constraint(
            "TSLA", ["AAPL"], self.returns, threshold=0.70
        )
        assert allowed is True
        assert corrs["AAPL"] < 0.70

    def test_highly_correlated_blocked(self):
        allowed, corrs = check_correlation_constraint(
            "MSFT", ["AAPL"], self.returns, threshold=0.70
        )
        assert allowed is False
        assert corrs["AAPL"] > 0.70

    def test_returns_correlation_values(self):
        _, corrs = check_correlation_constraint(
            "MSFT", ["AAPL", "TSLA"], self.returns, threshold=0.70
        )
        assert "AAPL" in corrs
        assert "TSLA" in corrs
        assert all(isinstance(v, float) for v in corrs.values())
        assert all(-1.0 <= v <= 1.0 for v in corrs.values())

    def test_symbol_not_in_data_returns_true(self):
        allowed, corrs = check_correlation_constraint(
            "UNKNOWN", ["AAPL"], self.returns, threshold=0.70
        )
        assert allowed is True

    def test_insufficient_data_returns_true(self):
        short_returns = self.returns.tail(10)
        allowed, _ = check_correlation_constraint(
            "MSFT", ["AAPL"], short_returns, lookback_days=60, threshold=0.70
        )
        assert allowed is True

    def test_custom_threshold(self):
        allowed, _ = check_correlation_constraint(
            "MSFT", ["AAPL"], self.returns, threshold=0.999
        )
        assert allowed is True


# ── apply_correlation_filter ──────────────────────────────────────────────────

class TestApplyCorrelationFilter:
    def setup_method(self):
        self.prices = _make_prices(80)
        n = 60
        dates = self.prices.index[-n:]
        entries = pd.DataFrame(False, index=dates, columns=["AAPL", "MSFT", "TSLA"])
        entries.iloc[10, entries.columns.get_loc("AAPL")] = True
        entries.iloc[10, entries.columns.get_loc("MSFT")] = True
        entries.iloc[20, entries.columns.get_loc("TSLA")] = True
        self.entries = entries
        self.oos_start = str(dates[0].date())
        self.oos_end = str(dates[-1].date())

    def test_reduce_mode_keeps_entries(self):
        cfg = _make_cfg(mode="reduce")
        filtered, size_mult, log = apply_correlation_filter(
            self.entries, self.prices, self.oos_start, self.oos_end, cfg
        )
        assert filtered.iloc[10]["AAPL"] == True
        assert filtered.iloc[10]["MSFT"] == True

    def test_reduce_mode_halves_size(self):
        cfg = _make_cfg(mode="reduce")
        _, size_mult, log = apply_correlation_filter(
            self.entries, self.prices, self.oos_start, self.oos_end, cfg
        )
        bar_date = self.entries.index[10]
        assert size_mult.loc[bar_date, "MSFT"] == 0.5

    def test_reject_mode_removes_lower_priority(self):
        cfg = _make_cfg(mode="reject")
        filtered, _, log = apply_correlation_filter(
            self.entries, self.prices, self.oos_start, self.oos_end, cfg
        )
        bar_date = self.entries.index[10]
        assert filtered.loc[bar_date, "AAPL"] == True
        assert not filtered.loc[bar_date, "MSFT"]

    def test_uncorrelated_tsla_unaffected(self):
        cfg = _make_cfg(mode="reject")
        filtered, size_mult, _ = apply_correlation_filter(
            self.entries, self.prices, self.oos_start, self.oos_end, cfg
        )
        bar_date = self.entries.index[20]
        assert filtered.loc[bar_date, "TSLA"] == True
        assert size_mult.loc[bar_date, "TSLA"] == 1.0

    def test_filter_log_records_events(self):
        cfg = _make_cfg(mode="reject")
        _, _, log = apply_correlation_filter(
            self.entries, self.prices, self.oos_start, self.oos_end, cfg
        )
        assert len(log) > 0
        assert any("CORR_REJECT" in entry or "CORR_REDUCE" in entry for entry in log)


# ── _build_symbol_to_sector ────────────────────────────────────────────────────

class TestBuildSymbolToSector:
    def test_basic_mapping(self):
        cfg = {"sector_caps": {"sectors": {"Tech": ["AAPL", "MSFT"], "ETF": ["SPY"]}}}
        mapping = _build_symbol_to_sector(cfg)
        assert mapping["AAPL"] == "Tech"
        assert mapping["MSFT"] == "Tech"
        assert mapping["SPY"] == "ETF"

    def test_empty_cfg_returns_empty(self):
        assert _build_symbol_to_sector({}) == {}

    def test_no_duplicates(self):
        cfg = {"sector_caps": {"sectors": {"A": ["AAPL", "MSFT"], "B": ["TSLA"]}}}
        mapping = _build_symbol_to_sector(cfg)
        assert len(mapping) == 3

    def test_unknown_symbol_not_in_mapping(self):
        cfg = {"sector_caps": {"sectors": {"Tech": ["AAPL"]}}}
        mapping = _build_symbol_to_sector(cfg)
        assert "UNKNOWN" not in mapping
