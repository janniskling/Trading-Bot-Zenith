"""Signal Filter Funnel diagnostics for the Zenith EMA-Momentum strategy.

Tracks how many candidate signals survive each filter step so you can see
exactly where the strategy loses trades — and flag over-restrictive filters.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from backtest.walk_forward import _ema, _rsi, _adx

# ── Thresholds ─────────────────────────────────────────────────────────────────

_MIN_TRADES_PER_YEAR = 2    # below → "Strategy too restrictive"
_MAX_TRADES_PER_YEAR = 50   # above → "Strategy too noisy"
_FILTER_KILL_THRESHOLD = 0.90  # warn when a single step drops >90% of survivors

# ── Per-symbol funnel ──────────────────────────────────────────────────────────

def run_filter_funnel(
    symbol: str,
    df: pd.DataFrame,
    spy_close: pd.Series,
    cfg: dict[str, Any],
    start: str | None = None,
    end: str | None = None,
) -> dict[str, int | str]:
    """Return cumulative filter-funnel counts for one symbol.

    Each step ANDs one more condition onto the previous survivors, so the
    numbers show how many trading days make it past each successive gate.
    """
    sc = cfg["strategy"]
    adx_lookback: int = sc.get("adx_lookback", 5)

    if start is not None or end is not None:
        df = df.loc[start:end]

    close = df["close"]
    spy_slice = spy_close.reindex(close.index)

    fast = _ema(close, sc["ema_fast"])
    mid  = _ema(close, sc["ema_mid"])
    slow = _ema(close, sc["ema_slow"])
    rsi  = _rsi(close, sc["rsi_period"])
    adx  = _adx(df,    sc["adx_period"])

    vol_ma = df["volume"].rolling(20).mean()

    spy_ema200   = _ema(spy_slice.ffill(), sc["spy_ema_trend"])
    spy_bull     = (spy_slice >= spy_ema200).fillna(False)

    ema_bull     = (fast > mid) & (mid > slow)
    cross_lookback = sc.get("crossover_lookback", 1)
    raw_cross    = (fast > mid) & (fast.shift(1) <= mid.shift(1))
    crossed      = raw_cross.rolling(cross_lookback).max().astype(bool)

    vol_ok       = df["volume"] > sc["volume_factor"] * vol_ma
    rsi_min      = sc.get("rsi_min", sc.get("rsi_oversold", 45))
    rsi_max      = sc.get("rsi_max", sc.get("rsi_overbought", 70))
    rsi_ok       = (rsi > rsi_min) & (rsi < rsi_max)
    adx_ok       = adx.rolling(adx_lookback).max() >= sc["adx_min"]
    bull_aligned = spy_bull

    # Cumulative steps
    s1 = crossed.fillna(False)
    s2 = s1 & ema_bull.fillna(False)
    s3 = s2 & vol_ok.fillna(False)
    s4 = s3 & rsi_ok.fillna(False)
    s5 = s4 & adx_ok.fillna(False)
    s6 = s5 & bull_aligned

    return {
        "symbol":             symbol,
        "trading_days":       len(df),
        "ema_crossover":      int(s1.sum()),
        "ema_crossover_bull": int(s2.sum()),
        "crossover_volume":   int(s3.sum()),
        "crossover_rsi":      int(s4.sum()),
        "crossover_adx":      int(s5.sum()),
        "final_entries":      int(s6.sum()),
    }


def run_all_funnels(
    ohlcv: dict[str, pd.DataFrame],
    spy_close: pd.Series,
    cfg: dict[str, Any],
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Run the filter funnel for every symbol and return a summary DataFrame."""
    rows = [
        run_filter_funnel(sym, df, spy_close, cfg, start=start, end=end)
        for sym, df in ohlcv.items()
    ]
    return pd.DataFrame(rows).set_index("symbol")


# ── Report printing ────────────────────────────────────────────────────────────

_STEP_LABELS: list[tuple[str, str]] = [
    ("ema_crossover",      "EMA-9 crossed above EMA-21"),
    ("ema_crossover_bull", "+ Price > EMA-50 (trend aligned)"),
    ("crossover_volume",   "+ Volume > {vf}× Avg(20)"),
    ("crossover_rsi",      "+ RSI in [{rlo}, {rhi}]"),
    ("crossover_adx",      "+ ADX ≥ {adxmin} (lookback {lb})"),
    ("final_entries",      "+ SPY > EMA-200 (macro filter)"),
]


def print_funnel_report(funnel_df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """Print an aggregated filter-funnel tree and flag over-restrictive steps."""
    sc = cfg["strategy"]
    subs = {
        "vf":    sc.get("volume_factor", 1.5),
        "rlo":   sc.get("rsi_min", sc.get("rsi_oversold", 45)),
        "rhi":   sc.get("rsi_max", sc.get("rsi_overbought", 70)),
        "adxmin": sc.get("adx_min", 25),
        "lb":    sc.get("adx_lookback", 5),
    }

    totals = funnel_df.sum(numeric_only=True)
    step_cols = [c for c, _ in _STEP_LABELS]

    lines = [
        "",
        "=" * 65,
        "SIGNAL FILTER FUNNEL  (aggregated across all symbols)",
        "=" * 65,
        f"  {'Trading Days':<44} {int(totals['trading_days']):>6}",
        "-" * 65,
    ]

    warnings: list[str] = []

    for i, (col, label_tpl) in enumerate(_STEP_LABELS):
        label = label_tpl.format(**subs)
        count = int(totals[col])

        # % of previous step (first step: % of trading days)
        prior = int(totals["trading_days"]) if i == 0 else int(totals[step_cols[i - 1]])
        pct_of_prior = count / prior if prior > 0 else 0.0
        eliminated   = 1.0 - pct_of_prior

        prefix = "├─" if i < len(_STEP_LABELS) - 1 else "└─"
        lines.append(
            f"  {prefix} {label:<42} {count:>6}  ({100*pct_of_prior:.0f}% of prev)"
        )

        if eliminated >= _FILTER_KILL_THRESHOLD and prior > 0:
            warnings.append(
                f"  ⚠️  '{label}' eliminates {100*eliminated:.0f}% of prior candidates "
                f"({prior} → {count}) — potential over-restriction"
            )

    lines.append("-" * 65)

    # Per-symbol detail for a handful of representative symbols
    highlights = ["NVDA", "AAPL", "SPY", "TSLA", "MSFT"]
    selected   = [s for s in highlights if s in funnel_df.index][:3]
    if selected:
        lines.append("")
        lines.append("  Per-symbol detail (sample):")
        col_names = ["Days", "Cross", "+Bull", "+Vol", "+RSI", "+ADX", "Final"]
        header = f"  {'Symbol':<8}" + "".join(f" {n:>7}" for n in col_names)
        lines.append(header)
        data_cols = ["trading_days", "ema_crossover", "ema_crossover_bull",
                     "crossover_volume", "crossover_rsi", "crossover_adx", "final_entries"]
        for sym in selected:
            row = funnel_df.loc[sym]
            vals = [int(row.get(c, 0)) for c in data_cols]
            lines.append(f"  {sym:<8}" + "".join(f" {v:>7}" for v in vals))

    lines.append("")
    lines.append("  Note: Correlation + sector-cap filters run per OOS window")
    lines.append("        (see filter_log.txt for per-window counts).")

    print("\n".join(lines))

    if warnings:
        print()
        for w in warnings:
            print(w)


# ── Signal frequency sanity check ─────────────────────────────────────────────

def check_signal_frequency(
    entries_df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> None:
    """Warn when signal rate is outside the healthy range for EMA crossover strategies.

    Expected range: 5–20 signals/year/symbol.
    Prints a summary but never raises — the caller decides whether to abort.
    """
    years = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days / 365.25
    per_sym        = entries_df.sum()
    per_sym_yr     = per_sym / years
    total          = per_sym.sum()
    total_yr       = total / years

    print(
        f"\n  Signal frequency  ({start_date} → {end_date}, "
        f"{years:.1f} yr, {len(entries_df.columns)} symbols)"
    )
    print(f"  Total signals : {int(total)}  ({total_yr:.1f}/yr across all symbols)")
    print(
        f"  Per symbol/yr : avg {per_sym_yr.mean():.1f}  "
        f"[min {per_sym_yr.min():.1f}, max {per_sym_yr.max():.1f}]"
    )

    too_few  = per_sym_yr[per_sym_yr < _MIN_TRADES_PER_YEAR]
    too_many = per_sym_yr[per_sym_yr > _MAX_TRADES_PER_YEAR]

    if not too_few.empty:
        syms = ", ".join(f"{s}({v:.1f})" for s, v in too_few.items())
        print(
            f"\n  ⚠️  STRATEGY TOO RESTRICTIVE — {len(too_few)} symbol(s) below "
            f"{_MIN_TRADES_PER_YEAR} signal/yr: {syms}"
        )
    if not too_many.empty:
        syms = ", ".join(f"{s}({v:.1f})" for s, v in too_many.items())
        print(
            f"\n  ⚠️  STRATEGY TOO NOISY — {len(too_many)} symbol(s) above "
            f"{_MAX_TRADES_PER_YEAR} signal/yr: {syms}"
        )
    if too_few.empty and too_many.empty:
        print("  ✅  Signal frequency looks healthy.")


# ── CSV export ─────────────────────────────────────────────────────────────────

def save_funnel_csv(funnel_df: pd.DataFrame, output_dir: str | Path) -> Path:
    """Write per-symbol funnel counts to signal_funnel.csv."""
    path = Path(output_dir) / "signal_funnel.csv"
    funnel_df.to_csv(path)
    return path
