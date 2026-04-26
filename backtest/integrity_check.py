"""Anti-curve-fitting guard for the Zenith backtest.

Every parameter change is logged here with its date, description, and
classification. The module enforces a hard limit on the number of changes
before requiring an explicit override flag — protecting against the most
common trap in strategy development: mistaking optimisation for discovery.

RULE: Add entries to PARAMETER_CHANGES. Never delete or edit existing ones.
      The log's immutability is what makes it trustworthy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ChangeType = Literal["bug-fix", "structural", "performance"]

# ── Hard limits (self-discipline thresholds) ──────────────────────────────────

_WARN_AT   = 4   # yellow: approaching curve-fitting territory
_RED_AT    = 5   # red: high risk, code comment with justification required
_HARD_STOP = 6   # --allow-curve-fitting flag required to proceed

# ── Expected performance ranges for a plain EMA-crossover strategy ────────────
# Source: academic literature on trend-following + Sharpe estimates for
# long-only EMA strategies on US large-cap equities (2010–2024).
# These are pre-specified BEFORE seeing results — changing them post-hoc is
# exactly the kind of thing this module exists to prevent.

EXPECTED_TOTAL_TRADES_MIN = 150
EXPECTED_TOTAL_TRADES_MAX = 400
EXPECTED_SHARPE_MIN  =  0.3
EXPECTED_SHARPE_MAX  =  1.0
EXPECTED_ALPHA_MIN   = -0.03   # -3% vs URTH
EXPECTED_ALPHA_MAX   =  0.03   # +3% vs URTH


# ── Canonical parameter change log ────────────────────────────────────────────

@dataclass(frozen=True)
class ParameterChange:
    change_id:   int
    date:        str
    description: str
    reason:      str
    change_type: ChangeType


PARAMETER_CHANGES: list[ParameterChange] = [
    ParameterChange(
        change_id=1,
        date="2026-04-25",
        description="adx_lookback: 1 (default, exact crossover bar) → 5",
        reason=(
            "ADX is Wilder's EWM — it lags by design. Requiring ADX ≥ 25 on the exact "
            "crossover bar produces a near-impossible condition: filter-funnel showed "
            "28 NVDA crossovers over 3 years, only 1 with ADX OK on the same day. "
            "Fix: check whether ADX reached the threshold within a rolling window. "
            "This is a structural logic conflict, not a performance optimisation."
        ),
        change_type="bug-fix",
    ),
    ParameterChange(
        change_id=2,
        date="2026-04-25",
        description="volume_factor: 1.5 → 1.2  |  adx_lookback: 5 → 10",
        reason=(
            "Filter-Funnel analysis showed volume_factor=1.5 eliminated 84% of valid "
            "crossover candidates (217 → 34 across 17 symbols, 6 years). For an "
            "EMA-crossover strategy, moderate above-average volume is sufficient — "
            "1.2× is the standard cited in O'Neil (CANSLIM) and Amibroker literature. "
            "adx_lookback=5 still too short: Wilder's own recommendation is 7–14 bars "
            "for ADX to stabilise after a regime change. Raised to 10 (midpoint). "
            "Neither change was driven by seeing OOS performance numbers first."
        ),
        change_type="structural",
    ),
    ParameterChange(
        change_id=3,
        date="2026-04-26",
        description=(
            "volume_factor: 1.2 → 1.1  |  adx_min: 25 → 20  |  "
            "rsi_min: 45 → 40  |  rsi_max: 70 → 75  |  crossover_lookback: 1 → 3  |  "
            "watchlist: 17 → ~80 symbols  |  max_positions_per_sector: 3 → 4  |  "
            "max_total_positions: 8 → 10"
        ),
        reason=(
            "Explicitly scoped as LEARNING-PROJECT MODE for 3-4 weeks of live observation. "
            "Statistical significance is NOT the goal — generating enough trades to "
            "observe the bot in action is. Justification per parameter: "
            "volume_factor=1.1 still above-average, not noise-level; "
            "adx_min=20 is Wilder's own original threshold (ADX was described as "
            "'trending' at 20+); rsi 40-75 includes weaker setups that are still "
            "directionally valid; crossover_lookback=3 is a realistic confirmation "
            "window (price can lag by 1-2 bars after signal). "
            "Watchlist expansion to ~80 symbols maximises signal frequency without "
            "changing the underlying strategy logic. "
            "These parameters will be reviewed after the 3-4 week period. "
            "Not driven by OOS performance numbers."
        ),
        change_type="structural",
    ),
]


# ── Main functions ─────────────────────────────────────────────────────────────

def check_integrity(allow_curve_fitting: bool = False) -> None:
    """Print the parameter change log and enforce the hard stop if triggered.

    Raises SystemExit if len(PARAMETER_CHANGES) >= _HARD_STOP and
    allow_curve_fitting is False.
    """
    n = len(PARAMETER_CHANGES)

    lines = [
        "",
        "=" * 65,
        "PARAMETER CHANGE LOG  (Anti-Curve-Fitting Guard)",
        "=" * 65,
        "Initial config (Strategy v1.0 — never seen OOS data):",
        "  volume_factor: 1.5  |  adx_min: 25  |  adx_lookback: 1",
        "",
    ]

    for ch in PARAMETER_CHANGES:
        type_label = {
            "bug-fix":     "BUG-FIX        (not performance-driven)",
            "structural":  "STRUCTURAL     (not performance-driven)",
            "performance": "PERFORMANCE OPTIMISATION  ← curve-fitting risk",
        }[ch.change_type]
        lines += [
            f"  Change {ch.change_id} ({ch.date}):",
            f"    What  : {ch.description}",
            f"    Type  : {type_label}",
            f"    Why   : {ch.reason}",
            "",
        ]

    lines.append(f"Total changes: {n} / {_HARD_STOP} hard limit  (warn at {_WARN_AT})")
    lines.append("-" * 65)

    # Status line
    if n < _WARN_AT:
        remaining = _WARN_AT - n
        lines.append(f"  Status: OK  ({remaining} change(s) before yellow-zone)")
    elif n == _WARN_AT:
        lines.append(
            f"  YELLOW ZONE: {n}/{_HARD_STOP} changes — approaching curve-fitting territory."
        )
        lines.append("  Next change requires strong structural justification.")
    elif n == _RED_AT:
        lines.append(
            f"  RED ZONE: {n}/{_HARD_STOP} changes — high curve-fitting risk."
        )
        lines.append("  A code comment with explicit justification is MANDATORY before proceeding.")
    elif n >= _HARD_STOP:
        hard_stop_msg = (
            f"HARD STOP: {n}/{_HARD_STOP} parameter changes reached.\n"
            f"Further parameter tuning at this point is more likely curve-fitting than\n"
            f"genuine discovery. If you believe the next change is justified:\n"
            f"  1. Write a detailed comment in config.yaml explaining WHY\n"
            f"  2. Re-run with: python run_backtest.py --allow-curve-fitting\n"
        )
        print("\n".join(lines))
        if not allow_curve_fitting:
            raise SystemExit(f"\n{'='*65}\n{hard_stop_msg}{'='*65}")
        lines.append("  OVERRIDE ACTIVE: --allow-curve-fitting passed. Proceeding.")

    print("\n".join(lines))


def print_expectation_check(
    total_trades: int,
    mean_sharpe: float,
    mean_alpha: float,
) -> None:
    """Print a pre-specified expectation check against actual backtest results.

    The expected ranges were set BEFORE running the backtest and are based on
    published academic baselines for long-only EMA strategies on US equities.
    Changing these ranges after seeing results would defeat the purpose.
    """

    def _row(label: str, actual: float, lo: float, hi: float, fmt: str = ".2f") -> str:
        actual_str = format(actual, fmt)
        range_str  = f"{format(lo, fmt)} – {format(hi, fmt)}"
        if lo <= actual <= hi:
            flag = "✅  Within range"
        elif actual > hi:
            flag = "⚠️  Suspiciously high — check for look-ahead bias"
        else:
            flag = "❌  Below expectations"
        return f"  {label:<22} actual={actual_str:<10} expected={range_str:<18} {flag}"

    trades_ok  = EXPECTED_TOTAL_TRADES_MIN <= total_trades  <= EXPECTED_TOTAL_TRADES_MAX
    sharpe_ok  = EXPECTED_SHARPE_MIN       <= mean_sharpe   <= EXPECTED_SHARPE_MAX
    alpha_ok   = EXPECTED_ALPHA_MIN        <= mean_alpha    <= EXPECTED_ALPHA_MAX
    all_ok     = trades_ok and sharpe_ok and alpha_ok
    any_high   = (total_trades > EXPECTED_TOTAL_TRADES_MAX
                  or mean_sharpe > EXPECTED_SHARPE_MAX
                  or mean_alpha  > EXPECTED_ALPHA_MAX)

    lines = [
        "",
        "=" * 65,
        "EXPECTATION CHECK  (ranges fixed before seeing results)",
        "=" * 65,
        _row("Total Trades",  total_trades,      EXPECTED_TOTAL_TRADES_MIN, EXPECTED_TOTAL_TRADES_MAX, ".0f"),
        _row("Mean Sharpe",   mean_sharpe,        EXPECTED_SHARPE_MIN,       EXPECTED_SHARPE_MAX),
        _row("Mean Alpha",    100 * mean_alpha,   100 * EXPECTED_ALPHA_MIN,  100 * EXPECTED_ALPHA_MAX,  ".1f"),
        "-" * 65,
    ]

    if all_ok:
        lines += [
            "  VERDICT: Within expectations.",
            "  Results are plausible for an EMA-crossover strategy. Proceed carefully;",
            "  statistical significance still needs Monte-Carlo validation before",
            "  drawing conclusions about real-world edge.",
        ]
    elif any_high:
        lines += [
            "  VERDICT: Suspiciously good.",
            "  Before celebrating: verify no look-ahead bias, check that OOS dates",
            "  don't overlap IS periods, and confirm vectorbt is not reusing signals.",
        ]
    else:
        lines += [
            "  VERDICT: Below expectations — no reliable edge detected.",
            "  This is the most likely outcome for any systematic strategy.",
            "  Options:",
            "    (A) Continue as paper-trading learning project (low capital)",
            "    (B) Redesign with fundamentally different signal logic",
            "    (C) Accept that passive MSCI World buy-and-hold wins",
            "  None of these is a failure — knowing what doesn't work is valuable.",
        ]

    lines.append("=" * 65)
    print("\n".join(lines))
