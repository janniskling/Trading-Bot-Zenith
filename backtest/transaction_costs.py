"""Realistic transaction cost model for the Zenith backtest.

Covers three cost layers:
  1. Slippage (spread + market impact)
  2. SEC Section 31 fee (sell-side only)
  3. FINRA TAF fee (sell-side only)
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ── Core slippage formula ──────────────────────────────────────────────────────

def calculate_slippage(
    price: float,
    volume: int,
    avg_daily_volume: int,
    bid_ask_spread_pct: float | None = None,
) -> float:
    """Compute realistic one-way slippage as a fraction of price.

    Components
    ----------
    base : 0.05% — irreducible crossing cost
    spread_cost : 0.5 × bid-ask spread (default 0.10% if spread unknown)
    market_impact : 0.10 × participation rate, only when trade > 1% of ADV

    Parameters
    ----------
    price : current price (not used directly but useful for callers)
    volume : shares in this trade
    avg_daily_volume : 20-day average daily volume in shares
    bid_ask_spread_pct : estimated bid-ask spread as fraction (e.g. 0.001 = 0.1%)

    Returns
    -------
    slippage as fraction (e.g. 0.0015 = 0.15%)
    """
    base = 0.0005  # 0.05%

    spread = bid_ask_spread_pct if bid_ask_spread_pct is not None else 0.001
    spread_cost = 0.5 * spread

    participation = volume / avg_daily_volume if avg_daily_volume > 0 else 0.0
    market_impact = 0.1 * participation if participation > 0.01 else 0.0

    total = base + spread_cost + market_impact
    return float(min(total, 0.02))  # cap at 2%


# ── Regulatory fees ────────────────────────────────────────────────────────────

def calculate_regulatory_fees(
    price: float,
    qty: int,
    side: str,
    sec_fee_rate: float = 0.0000229,
    taf_fee_per_share: float = 0.000166,
    taf_fee_max: float = 8.30,
) -> float:
    """SEC + FINRA TAF fees as a fraction of trade value (sell-side only).

    SEC fee : 0.00229% × sell value
    TAF fee : $0.000166 × shares, capped at $8.30

    Returns
    -------
    fee as fraction of trade value (0.0 for buy-side)
    """
    if side.lower() != "sell":
        return 0.0

    trade_value = price * qty
    if trade_value <= 0:
        return 0.0

    sec_fee = sec_fee_rate * trade_value
    taf_fee = min(taf_fee_per_share * qty, taf_fee_max)
    return (sec_fee + taf_fee) / trade_value


# ── Convenience: round-trip cost ──────────────────────────────────────────────

def round_trip_cost(
    price: float,
    qty: int,
    avg_daily_volume: int,
    bid_ask_spread_pct: float | None = None,
) -> float:
    """Total cost fraction for one full buy + sell round trip."""
    slip = calculate_slippage(price, qty, avg_daily_volume, bid_ask_spread_pct)
    fees = calculate_regulatory_fees(price, qty, "sell")
    return slip * 2 + fees  # slippage on both legs


# ── Vectorised version for the backtest ───────────────────────────────────────

def build_cost_slippage_df(
    ohlcv: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Build a per-bar, per-symbol slippage fraction DataFrame.

    Uses the transaction_costs model: base + spread proxy + market impact.
    The result replaces the naive high-low spread estimator used in baseline mode.
    """
    tc = cfg["transaction_costs"]
    base: float = tc["slippage_base"]
    spread_default: float = tc["spread_default"]
    spread_factor: float = tc["spread_factor"]
    impact_threshold: float = tc["market_impact_threshold"]
    impact_factor: float = tc["market_impact_factor"]

    result: dict[str, pd.Series] = {}
    for sym, df in ohlcv.items():
        # Bid-ask spread proxy from intraday range
        raw_spread = (df["high"] - df["low"]) / df["close"]
        spread_cost = (spread_factor * raw_spread).clip(lower=spread_factor * spread_default)

        # Market impact: bar volume vs 20-day rolling average
        avg_vol = df["volume"].rolling(20, min_periods=5).mean()
        participation = df["volume"] / avg_vol.replace(0.0, float("nan"))
        market_impact = (impact_factor * participation).where(
            participation > impact_threshold, 0.0
        ).fillna(0.0)

        total = (base + spread_cost + market_impact).clip(upper=0.02)
        result[sym] = total

    return pd.DataFrame(result)


def regulatory_fee_fraction(cfg: dict[str, Any]) -> float:
    """Approximate sell-side regulatory fees as a single scalar (fraction of value).

    Used to set vectorbt's `fees` parameter. Since vectorbt applies fees to
    both sides symmetrically, we use half the sell-side value.
    """
    tc = cfg["transaction_costs"]
    # Estimate on a representative $5,000 trade of 100 shares at $50
    representative_price = 50.0
    representative_qty = 100
    fee = calculate_regulatory_fees(
        representative_price,
        representative_qty,
        "sell",
        sec_fee_rate=tc["sec_fee_rate"],
        taf_fee_per_share=tc["taf_fee_per_share"],
        taf_fee_max=tc["taf_fee_max"],
    )
    return fee / 2  # halved because vectorbt applies to both buy and sell
