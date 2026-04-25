"""Unit tests for backtest/correlation_filter.py"""
import numpy as np
import pandas as pd

from backtest.correlation_filter import (
    check_correlation_constraint,
    apply_correlation_filter,
    SYMBOL_TO_SECTOR,
    SECTOR_MAP,
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


# ── check_correlation_constraint ──────────────────────────────────────────────

class TestCheckCorrelationConstraint:
    def setup_method(self):
        self.returns = _make_returns()

    def test_no_open_positions_always_allowed(self):
        allowed, corrs = check_correlation_constraint("AAPL", [], self.returns)
        assert allowed is True
        assert corrs == {}

    def test_uncorrelated_symbol_allowed(self):
        # TSLA is independent → should be allowed even with AAPL open
        allowed, corrs = check_correlation_constraint(
            "TSLA", ["AAPL"], self.returns, threshold=0.70
        )
        assert allowed is True
        assert corrs["AAPL"] < 0.70

    def test_highly_correlated_blocked(self):
        # MSFT is ~0.95 correlated with AAPL → should be blocked
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
        assert allowed is True  # not enough data → don't block

    def test_custom_threshold(self):
        # With threshold=0.999 the synthetic data (corr ~0.997) should pass
        allowed, _ = check_correlation_constraint(
            "MSFT", ["AAPL"], self.returns, threshold=0.999
        )
        assert allowed is True


# ── apply_correlation_filter ──────────────────────────────────────────────────

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
            "sectors": SECTOR_MAP,
        },
    }


class TestApplyCorrelationFilter:
    def setup_method(self):
        self.prices = _make_prices(80)
        n = 60
        dates = self.prices.index[-n:]
        # Both AAPL and MSFT have entry signals on the same bar
        entries = pd.DataFrame(False, index=dates, columns=["AAPL", "MSFT", "TSLA"])
        entries.iloc[10, entries.columns.get_loc("AAPL")] = True
        entries.iloc[10, entries.columns.get_loc("MSFT")] = True  # same bar as AAPL
        entries.iloc[20, entries.columns.get_loc("TSLA")] = True
        self.entries = entries
        self.oos_start = str(dates[0].date())
        self.oos_end = str(dates[-1].date())

    def test_reduce_mode_keeps_entries(self):
        cfg = _make_cfg(mode="reduce")
        filtered, size_mult, log = apply_correlation_filter(
            self.entries, self.prices, self.oos_start, self.oos_end, cfg
        )
        # Both entries should survive in reduce mode
        assert filtered.iloc[10]["AAPL"] is True or filtered.iloc[10]["AAPL"] == True
        assert filtered.iloc[10]["MSFT"] is True or filtered.iloc[10]["MSFT"] == True

    def test_reduce_mode_halves_size(self):
        cfg = _make_cfg(mode="reduce")
        _, size_mult, log = apply_correlation_filter(
            self.entries, self.prices, self.oos_start, self.oos_end, cfg
        )
        bar_date = self.entries.index[10]
        # Lower-priority symbol (MSFT, alphabetically after AAPL) should be halved
        assert size_mult.loc[bar_date, "MSFT"] == 0.5

    def test_reject_mode_removes_lower_priority(self):
        cfg = _make_cfg(mode="reject")
        filtered, _, log = apply_correlation_filter(
            self.entries, self.prices, self.oos_start, self.oos_end, cfg
        )
        bar_date = self.entries.index[10]
        # AAPL survives (higher priority alphabetically), MSFT is blocked
        assert filtered.loc[bar_date, "AAPL"] is True or filtered.loc[bar_date, "AAPL"] == True
        assert not filtered.loc[bar_date, "MSFT"]

    def test_uncorrelated_tsla_unaffected(self):
        cfg = _make_cfg(mode="reject")
        filtered, size_mult, _ = apply_correlation_filter(
            self.entries, self.prices, self.oos_start, self.oos_end, cfg
        )
        bar_date = self.entries.index[20]
        assert filtered.loc[bar_date, "TSLA"] is True or filtered.loc[bar_date, "TSLA"] == True
        assert size_mult.loc[bar_date, "TSLA"] == 1.0

    def test_filter_log_records_events(self):
        cfg = _make_cfg(mode="reject")
        _, _, log = apply_correlation_filter(
            self.entries, self.prices, self.oos_start, self.oos_end, cfg
        )
        assert len(log) > 0
        assert any("CORR_REJECT" in entry or "CORR_REDUCE" in entry for entry in log)


# ── Sector mapping ─────────────────────────────────────────────────────────────

class TestSectorMapping:
    def test_all_watchlist_symbols_mapped(self):
        watchlist = ["AAPL","MSFT","NVDA","TSLA","AMZN","META","GOOGL","AMD","CRM",
                     "PYPL","INTC","SOFI","SNAP","SPY","QQQ","IWM","SYM"]
        unmapped = [s for s in watchlist if s not in SYMBOL_TO_SECTOR]
        assert unmapped == [], f"Unmapped symbols: {unmapped}"

    def test_no_duplicate_symbols_across_sectors(self):
        all_syms = [sym for syms in SECTOR_MAP.values() for sym in syms]
        assert len(all_syms) == len(set(all_syms)), "Duplicate symbols across sectors"

    def test_etf_sector_correct(self):
        assert SYMBOL_TO_SECTOR["SPY"] == "ETF"
        assert SYMBOL_TO_SECTOR["QQQ"] == "ETF"
        assert SYMBOL_TO_SECTOR["IWM"] == "ETF"
