"""Correlation- and sector-based portfolio construction filters.

Two independent constraints:
  1. Pairwise correlation filter — don't pile into positions that move together.
  2. Sector cap — max N positions per GICS sector simultaneously.

Both can be used standalone (live trading via check_correlation_constraint) or
applied to a vectorbt signal DataFrame before running the backtest.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _build_symbol_to_sector(cfg: dict) -> dict[str, str]:
    """Build symbol→sector lookup from cfg['sector_caps']['sectors']."""
    sectors = cfg.get("sector_caps", {}).get("sectors", {})
    return {
        sym: sector
        for sector, syms in sectors.items()
        for sym in (syms or [])
    }


# ── Live-trading helper ────────────────────────────────────────────────────────

def check_correlation_constraint(
    new_symbol: str,
    open_positions: list[str],
    returns_data: pd.DataFrame,
    lookback_days: int = 60,
    threshold: float = 0.70,
) -> tuple[bool, dict[str, float]]:
    """Check whether adding new_symbol violates the correlation constraint.

    Parameters
    ----------
    new_symbol : ticker to evaluate
    open_positions : tickers already in the portfolio
    returns_data : wide DataFrame of daily returns (date × symbol)
    lookback_days : rolling window for Pearson correlation
    threshold : max allowed pairwise correlation

    Returns
    -------
    (is_allowed, correlations_dict)
        is_allowed  — True if the trade may proceed (at full size)
        correlations_dict — {symbol: correlation_value} for all open positions
    """
    if not open_positions or new_symbol not in returns_data.columns:
        return True, {}

    # Take the most recent `lookback_days` rows
    recent = returns_data[[new_symbol] + [s for s in open_positions if s in returns_data.columns]]
    recent = recent.dropna().tail(lookback_days)

    if len(recent) < 20:
        return True, {}

    correlations: dict[str, float] = {}
    new_returns = recent[new_symbol]
    for sym in open_positions:
        if sym not in recent.columns:
            continue
        corr = float(new_returns.corr(recent[sym]))
        if np.isfinite(corr):
            correlations[sym] = round(corr, 4)

    max_corr = max(correlations.values(), default=0.0)
    is_allowed = max_corr <= threshold
    return is_allowed, correlations


# ── Backtest signal processing ─────────────────────────────────────────────────

def _compute_avg_correlation_matrix(
    prices: pd.DataFrame,
    oos_start: str,
    oos_end: str,
    lookback_days: int,
) -> pd.DataFrame:
    """Pearson correlation of daily returns over the OOS window."""
    slice_ = prices.loc[oos_start:oos_end]
    if len(slice_) < lookback_days:
        slice_ = prices.tail(lookback_days * 2)
    returns = slice_.pct_change().dropna()
    return returns.corr()


def apply_correlation_filter(
    entries: pd.DataFrame,
    prices: pd.DataFrame,
    oos_start: str,
    oos_end: str,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Vectorised correlation filter applied to an entry signal DataFrame.

    For each bar where multiple symbols fire simultaneously, correlated pairs
    (corr > threshold) are handled by:
      - mode='reduce' : halve the position size of the lower-priority symbol
      - mode='reject' : zero out the entry for the lower-priority symbol

    Priority is determined by alphabetical order (deterministic, repeatable).

    Parameters
    ----------
    entries : bool DataFrame (date × symbol) — raw entry signals
    prices : close price DataFrame used to compute returns
    oos_start / oos_end : date strings for the OOS window
    cfg : full config dict (uses cfg['correlation'] and cfg['sector_caps'])

    Returns
    -------
    filtered_entries : bool DataFrame (entries after sector cap + correlation filter)
    size_multipliers : float DataFrame (1.0 normally, 0.5 for reduced positions)
    filter_log : list of human-readable strings describing each filtered trade
    """
    corr_cfg = cfg["correlation"]
    threshold: float = corr_cfg["threshold"]
    lookback: int = corr_cfg["lookback_days"]
    mode: str = corr_cfg["mode"]
    reduction: float = corr_cfg["size_reduction"]

    sc_cfg = cfg["sector_caps"]
    max_per_sector: int = sc_cfg["max_per_sector"]
    max_etf: int = sc_cfg["max_etf"]

    corr_matrix = _compute_avg_correlation_matrix(prices, oos_start, oos_end, lookback)
    symbol_to_sector = _build_symbol_to_sector(cfg)

    filtered = entries.copy().astype(bool)
    size_mult = pd.DataFrame(1.0, index=entries.index, columns=entries.columns)
    filter_log: list[str] = []

    for bar_date, row in entries.iterrows():
        mask = row.fillna(False).astype(bool)
        active = sorted(mask[mask].index.tolist())  # symbols with signals on this bar
        if len(active) < 2:
            continue

        # ── 1. Sector cap ──────────────────────────────────────────────────
        sector_counts: dict[str, int] = {}
        allowed_by_sector: list[str] = []

        for sym in active:
            sector = symbol_to_sector.get(sym, "Other")
            cap = max_etf if sector == "ETF" else max_per_sector
            count = sector_counts.get(sector, 0)
            if count < cap:
                sector_counts[sector] = count + 1
                allowed_by_sector.append(sym)
            else:
                filtered.loc[bar_date, sym] = False
                filter_log.append(
                    f"{bar_date} | SECTOR_CAP  | {sym} blocked "
                    f"(sector={sector} already at cap={cap})"
                )

        active = allowed_by_sector
        if len(active) < 2:
            continue

        # ── 2. Pairwise correlation filter ────────────────────────────────
        # Build sorted list; iterate pairs, handle lower-priority symbol
        processed: set[str] = set()
        for i, sym_a in enumerate(active):
            for sym_b in active[i + 1:]:
                if sym_b in processed:
                    continue
                corr = corr_matrix.loc[sym_a, sym_b] if (
                    sym_a in corr_matrix.index and sym_b in corr_matrix.columns
                ) else 0.0
                if not np.isfinite(corr) or corr <= threshold:
                    continue

                # sym_b is lower priority (comes later alphabetically)
                if mode == "reject":
                    if filtered.loc[bar_date, sym_b]:
                        filtered.loc[bar_date, sym_b] = False
                        processed.add(sym_b)
                        filter_log.append(
                            f"{bar_date} | CORR_REJECT | {sym_b} blocked "
                            f"(corr={corr:.2f} with {sym_a} > {threshold})"
                        )
                else:  # reduce
                    if size_mult.loc[bar_date, sym_b] == 1.0:
                        size_mult.loc[bar_date, sym_b] = reduction
                        processed.add(sym_b)
                        filter_log.append(
                            f"{bar_date} | CORR_REDUCE | {sym_b} halved "
                            f"(corr={corr:.2f} with {sym_a} > {threshold})"
                        )

    return filtered, size_mult, filter_log


def summarise_filter_log(filter_log: list[str]) -> str:
    """Human-readable summary of what was filtered and why."""
    if not filter_log:
        return "No trades filtered."

    sector_blocks = sum(1 for l in filter_log if "SECTOR_CAP" in l)
    corr_rejects = sum(1 for l in filter_log if "CORR_REJECT" in l)
    corr_reduces = sum(1 for l in filter_log if "CORR_REDUCE" in l)

    lines = [
        f"Total filter events : {len(filter_log)}",
        f"  Sector cap blocks : {sector_blocks}",
        f"  Correlation rejects: {corr_rejects}",
        f"  Correlation reduces: {corr_reduces}",
    ]
    return "\n".join(lines)
