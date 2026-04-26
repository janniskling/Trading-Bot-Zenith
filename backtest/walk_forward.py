"""Walk-Forward Validation for the Zenith EMA-Momentum strategy.

Supports three backtest configurations:
  Baseline          — no costs, no filters (best-case)
  +Costs            — realistic slippage + regulatory fees
  +Costs+Correlation — costs + pairwise correlation filter + sector cap
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt
import yaml
import yfinance as yf
from dateutil.relativedelta import relativedelta

from backtest.transaction_costs import build_cost_slippage_df, regulatory_fee_fraction
from backtest.correlation_filter import apply_correlation_filter, summarise_filter_log

warnings.filterwarnings("ignore", category=FutureWarning)

_CFG_PATH = Path(__file__).parent / "config.yaml"


# ── Config loading ─────────────────────────────────────────────────────────────

def load_config(path: Path = _CFG_PATH) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def flatten_watchlist(cfg: dict[str, Any]) -> list[str]:
    """Return an ordered, deduplicated flat list of symbols from the sectored watchlist."""
    watchlist = cfg.get("watchlist", {})
    if not watchlist:
        return cfg.get("data", {}).get("symbols", [])
    seen: dict[str, None] = {}
    for syms in watchlist.values():
        for s in (syms or []):
            seen[s] = None
    return list(seen)


# ── Backtest configuration ─────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    """Controls which cost and filter layers are active."""
    name: str = "Baseline"
    use_costs: bool = False        # realistic slippage + regulatory fees
    use_correlation: bool = False  # correlation filter + sector cap


CONFIGS = [
    BacktestConfig("Baseline",              use_costs=False, use_correlation=False),
    BacktestConfig("+Costs",                use_costs=True,  use_correlation=False),
    BacktestConfig("+Costs+Correlation",    use_costs=True,  use_correlation=True),
]


# ── Data download ──────────────────────────────────────────────────────────────

def download_ohlcv(symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    raw = yf.download(
        symbols,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    result: dict[str, pd.DataFrame] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for sym in symbols:
            try:
                df = raw.xs(sym, axis=1, level=1).copy()
                df.columns = [c.lower() for c in df.columns]
                df = df.dropna()
                if len(df) > 50:
                    result[sym] = df
            except KeyError:
                pass
    else:
        sym = symbols[0]
        df = raw.copy()
        df.columns = [c.lower() for c in df.columns]
        result[sym] = df
    return result


# ── Technical indicators ───────────────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    ph, pl, pc = h.shift(1), l.shift(1), c.shift(1)
    up_move = h - ph
    down_move = pl - l
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr_s = tr.ewm(com=period - 1, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=c.index).ewm(com=period - 1, min_periods=period).mean() / atr_s
    minus_di = 100 * pd.Series(minus_dm, index=c.index).ewm(com=period - 1, min_periods=period).mean() / atr_s
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.ewm(com=period - 1, min_periods=period).mean()


def compute_signals(
    ohlcv: dict[str, pd.DataFrame],
    spy_close: pd.Series,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute entry/exit signals and ATR-based stop fractions."""
    sc = cfg["strategy"]
    spy_ema200 = _ema(spy_close, sc["spy_ema_trend"])
    spy_bull = spy_close >= spy_ema200

    all_entries: dict[str, pd.Series] = {}
    all_exits: dict[str, pd.Series] = {}
    all_sl_fracs: dict[str, pd.Series] = {}

    for sym, df in ohlcv.items():
        close = df["close"]
        fast = _ema(close, sc["ema_fast"])
        mid = _ema(close, sc["ema_mid"])
        slow = _ema(close, sc["ema_slow"])
        rsi = _rsi(close, sc["rsi_period"])
        adx = _adx(df, sc["adx_period"])
        atr = _atr(df, sc["atr_period"])

        vol_ma = df["volume"].rolling(20).mean()
        vol_ok = df["volume"] > sc["volume_factor"] * vol_ma
        ema_bull = (fast > mid) & (mid > slow)

        # Crossover within the last crossover_lookback bars
        cross_lookback: int = sc.get("crossover_lookback", 1)
        raw_cross = (fast > mid) & (fast.shift(1) <= mid.shift(1))
        crossed = raw_cross.rolling(cross_lookback).max().astype(bool)

        rsi_min = sc.get("rsi_min", sc.get("rsi_oversold", 45))
        rsi_max = sc.get("rsi_max", sc.get("rsi_overbought", 70))
        rsi_ok = (rsi > rsi_min) & (rsi < rsi_max)
        adx_lookback: int = sc.get("adx_lookback", 5)
        adx_ok = adx.rolling(adx_lookback).max() >= sc["adx_min"]
        bull_aligned = spy_bull.reindex(close.index).fillna(False)

        entry = (crossed & ema_bull & rsi_ok & adx_ok & vol_ok & bull_aligned).fillna(False)
        atr_frac = (sc["atr_stop_multiplier"] * atr / close).clip(upper=0.15)
        exit_sig = (fast < mid).fillna(False)

        all_entries[sym] = entry.astype(bool)
        all_exits[sym] = exit_sig.astype(bool)
        all_sl_fracs[sym] = atr_frac

    syms = list(all_entries.keys())
    return (
        pd.DataFrame(all_entries)[syms],
        pd.DataFrame(all_exits)[syms],
        pd.DataFrame(all_sl_fracs)[syms],
    )


def build_baseline_slippage_df(
    ohlcv: dict[str, pd.DataFrame], cfg: dict[str, Any]
) -> pd.DataFrame:
    """Naive high-low spread estimator — used for the Baseline config."""
    sc = cfg["slippage"]
    result: dict[str, pd.Series] = {}
    for sym, df in ohlcv.items():
        raw = (df["high"] - df["low"]) / df["close"]
        result[sym] = (raw * sc["spread_factor"]).clip(
            lower=sc["min_pct"], upper=sc["max_pct"]
        )
    return pd.DataFrame(result)


# ── Walk-Forward Windows ───────────────────────────────────────────────────────

@dataclass
class WFWindow:
    is_start: date
    is_end: date
    oos_start: date
    oos_end: date
    window_id: int


def generate_windows(cfg: dict[str, Any]) -> list[WFWindow]:
    wf = cfg["walk_forward"]
    universe_start = pd.Timestamp(cfg["data"]["start_date"])
    is_start = universe_start + relativedelta(months=12)
    total_end = pd.Timestamp(cfg["data"]["end_date"])

    windows: list[WFWindow] = []
    wid = 1
    while True:
        is_end = is_start + relativedelta(months=wf["in_sample_months"]) - relativedelta(days=1)
        oos_start = is_end + relativedelta(days=1)
        oos_end = oos_start + relativedelta(months=wf["out_of_sample_months"]) - relativedelta(days=1)
        if oos_end > total_end:
            break
        windows.append(WFWindow(is_start.date(), is_end.date(), oos_start.date(), oos_end.date(), wid))
        is_start += relativedelta(months=wf["stride_months"])
        wid += 1
    return windows


# ── Metrics ────────────────────────────────────────────────────────────────────

@dataclass
class WindowMetrics:
    window_id: int
    oos_start: date
    oos_end: date
    total_return: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    benchmark_return: float
    alpha: float


def _safe(val: Any, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def compute_metrics(
    pf: vbt.Portfolio,
    window: WFWindow,
    benchmark_oos: pd.Series,
) -> WindowMetrics:
    # group_by=True aggregates all symbol columns into one portfolio view,
    # avoiding the silent mean-across-columns that int() would truncate to 0.
    stats = pf.stats(group_by=True)
    total_return = _safe(stats.get("Total Return [%]", 0)) / 100
    sharpe = _safe(stats.get("Sharpe Ratio", 0))
    sortino = _safe(stats.get("Sortino Ratio", 0))
    calmar = _safe(stats.get("Calmar Ratio", 0))
    max_dd = _safe(stats.get("Max Drawdown [%]", 0)) / 100
    win_rate = _safe(stats.get("Win Rate [%]", 0)) / 100
    profit_factor = _safe(stats.get("Profit Factor", 0))
    num_trades = int(_safe(stats.get("Total Trades", 0)))

    bm_slice = benchmark_oos.loc[str(window.oos_start): str(window.oos_end)]
    bm_return = float((bm_slice.iloc[-1] / bm_slice.iloc[0]) - 1) if len(bm_slice) >= 2 else 0.0
    alpha = total_return - bm_return

    return WindowMetrics(
        window_id=window.window_id,
        oos_start=window.oos_start,
        oos_end=window.oos_end,
        total_return=total_return,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=max_dd,
        win_rate=win_rate,
        profit_factor=profit_factor,
        num_trades=num_trades,
        benchmark_return=bm_return,
        alpha=alpha,
    )


# ── Per-window OOS backtest ────────────────────────────────────────────────────

def run_oos_backtest(
    window: WFWindow,
    prices: pd.DataFrame,
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    sl_fracs: pd.DataFrame,
    slippage_df: pd.DataFrame,
    benchmark_prices: pd.Series,
    cfg: dict[str, Any],
    backtest_cfg: BacktestConfig | None = None,
) -> tuple[WindowMetrics | None, list[str]]:
    """Run a single OOS backtest window.

    Returns
    -------
    (WindowMetrics | None, filter_log)
    """
    if backtest_cfg is None:
        backtest_cfg = BacktestConfig()

    oos_slice = slice(str(window.oos_start), str(window.oos_end))
    price_oos = prices.loc[oos_slice]
    entry_oos = entries.loc[oos_slice]
    exit_oos = exits.loc[oos_slice]
    sl_oos = sl_fracs.loc[oos_slice]
    slip_oos = slippage_df.loc[oos_slice]

    cols = [c for c in price_oos.columns if c in entry_oos.columns]
    if not cols:
        return None, []

    price_oos = price_oos[cols]
    entry_oos = entry_oos[cols]
    exit_oos = exit_oos[cols]
    sl_oos = sl_oos[cols]
    slip_oos = slip_oos[cols]

    filter_log: list[str] = []

    # Base size DataFrame (uniform 1% risk per trade)
    rc = cfg["risk"]
    size_df = pd.DataFrame(
        rc["risk_per_trade_pct"],
        index=price_oos.index,
        columns=cols,
    )

    # Correlation + sector cap filter
    if backtest_cfg.use_correlation:
        entry_oos, size_mult, filter_log = apply_correlation_filter(
            entries=entry_oos,
            prices=price_oos,
            oos_start=str(window.oos_start),
            oos_end=str(window.oos_end),
            cfg=cfg,
        )
        size_df = size_df * size_mult

    # Slippage: costs model or baseline
    fees = regulatory_fee_fraction(cfg) if backtest_cfg.use_costs else 0.0
    if backtest_cfg.use_costs:
        avg_slip = slip_oos.median().fillna(0.001)
        slip_vals = {col: avg_slip.get(col, 0.001) for col in cols}
        effective_slip = pd.DataFrame(
            {col: slip_vals[col] for col in cols},
            index=price_oos.index,
        )
    else:
        effective_slip = 0.0

    # Ensure numpy dtypes — numba rejects Python object arrays
    entry_oos = entry_oos.astype(bool)
    exit_oos = exit_oos.astype(bool)
    sl_oos = sl_oos.astype(float).fillna(0.0)
    price_oos = price_oos.astype(float)

    try:
        pf = vbt.Portfolio.from_signals(
            close=price_oos,
            entries=entry_oos,
            exits=exit_oos,
            sl_stop=sl_oos,
            init_cash=10_000.0,
            size=size_df,
            size_type="Percent",
            fees=fees,
            slippage=effective_slip,
            accumulate=False,
            freq="D",
        )
    except Exception as exc:
        print(f"  [window {window.window_id}] vectorbt error: {exc}")
        return None, filter_log

    metrics = compute_metrics(pf, window, benchmark_prices)
    return metrics, filter_log


# ── Aggregation & Verdict ──────────────────────────────────────────────────────

@dataclass
class WalkForwardResult:
    name: str
    windows: list[WindowMetrics]
    metrics_df: pd.DataFrame
    summary: dict[str, float] = field(default_factory=dict)
    verdict: str = ""
    filter_log: list[str] = field(default_factory=list)

    def equity_curve(self) -> pd.Series:
        """Compounded OOS equity curve (start = 1.0) from per-window returns."""
        data: dict[pd.Timestamp, float] = {}
        equity = 1.0
        for m in sorted(self.windows, key=lambda x: x.oos_start):
            data[pd.Timestamp(m.oos_start)] = equity
            equity *= (1 + m.total_return)
        if self.windows:
            last = max(self.windows, key=lambda x: x.oos_end)
            data[pd.Timestamp(last.oos_end)] = equity
        return pd.Series(data)

    def benchmark_curve(self) -> pd.Series:
        """Compounded benchmark OOS equity curve (start = 1.0)."""
        data: dict[pd.Timestamp, float] = {}
        equity = 1.0
        for m in sorted(self.windows, key=lambda x: x.oos_start):
            data[pd.Timestamp(m.oos_start)] = equity
            equity *= (1 + m.benchmark_return)
        if self.windows:
            last = max(self.windows, key=lambda x: x.oos_end)
            data[pd.Timestamp(last.oos_end)] = equity
        return pd.Series(data)


@dataclass
class ComparisonResult:
    baseline: WalkForwardResult
    with_costs: WalkForwardResult
    with_costs_and_correlation: WalkForwardResult

    def configs(self) -> list[WalkForwardResult]:
        return [self.baseline, self.with_costs, self.with_costs_and_correlation]


def _aggregate(results: list[WindowMetrics]) -> dict[str, float]:
    df = pd.DataFrame([r.__dict__ for r in results])
    return {
        "mean_return": df["total_return"].mean(),
        "mean_sharpe": df["sharpe"].mean(),
        "mean_sortino": df["sortino"].mean(),
        "mean_calmar": df["calmar"].mean(),
        "mean_max_dd": df["max_drawdown"].mean(),
        "mean_win_rate": df["win_rate"].mean(),
        "mean_profit_factor": df["profit_factor"].mean(),
        "mean_alpha": df["alpha"].mean(),
        "beat_rate": (df["alpha"] > 0).mean(),
        "positive_return_rate": (df["total_return"] > 0).mean(),
        "total_windows": len(df),
        "total_trades": df["num_trades"].sum(),
    }


def _verdict(summary: dict[str, float], name: str) -> str:
    beat = summary["beat_rate"]
    sharpe = summary["mean_sharpe"]
    alpha = summary["mean_alpha"]

    lines = [f"{'='*55}", f"VERDICT — {name}", f"{'='*55}"]
    if beat >= 0.65 and sharpe >= 0.5 and alpha > 0.02:
        lines += ["✅  STRATEGY HOLDS OUT-OF-SAMPLE",
                  f"    Beat benchmark in {beat:.0%} of windows",
                  f"    Mean Sharpe {sharpe:.2f} | Mean alpha {alpha*100:+.1f}%/window"]
    elif beat >= 0.50:
        lines += ["⚠️   STRATEGY MARGINALLY VIABLE",
                  f"    Beat benchmark in {beat:.0%} of windows",
                  "    Monitor closely before scaling."]
    else:
        lines += ["❌  STRATEGY UNDERPERFORMS BUY-AND-HOLD URTH",
                  f"    Beat benchmark in only {beat:.0%} of windows",
                  f"    Mean alpha {alpha*100:+.1f}% — no clear edge.",
                  "    Do NOT deploy without substantial improvements."]
    lines += [
        "",
        f"  Windows : {summary['total_windows']:.0f}  |  "
        f"Positive return: {summary['positive_return_rate']:.0%}",
        f"  Sharpe  : {summary['mean_sharpe']:.2f}  |  "
        f"Sortino: {summary['mean_sortino']:.2f}  |  "
        f"Calmar: {summary['mean_calmar']:.2f}",
        f"  Max DD  : {summary['mean_max_dd']*100:.1f}%  |  "
        f"Win Rate: {summary['mean_win_rate']:.0%}  |  "
        f"Profit Factor: {summary['mean_profit_factor']:.2f}",
        f"{'='*55}",
    ]
    return "\n".join(lines)


def _comparison_verdict(
    baseline: WalkForwardResult,
    costs: WalkForwardResult,
    full: WalkForwardResult,
) -> str:
    b_alpha = baseline.summary["mean_alpha"]
    c_alpha = costs.summary["mean_alpha"]
    f_alpha = full.summary["mean_alpha"]

    cost_drag = b_alpha - c_alpha
    cost_pct = (cost_drag / abs(b_alpha) * 100) if b_alpha != 0 else float("inf")
    dd_reduction = (costs.summary["mean_max_dd"] - full.summary["mean_max_dd"]) / abs(costs.summary["mean_max_dd"]) if costs.summary["mean_max_dd"] != 0 else 0
    trade_reduction = (costs.summary["total_trades"] - full.summary["total_trades"]) / max(costs.summary["total_trades"], 1) * 100

    lines = [
        "",
        f"{'='*65}",
        "3-CONFIGURATION COMPARISON SUMMARY",
        f"{'='*65}",
        f"{'Metric':<22} {'Baseline':>10} {'  +Costs':>10} {'+C+Corr':>10}",
        f"{'-'*65}",
        f"{'Mean Sharpe':<22} {baseline.summary['mean_sharpe']:>10.2f} {costs.summary['mean_sharpe']:>10.2f} {full.summary['mean_sharpe']:>10.2f}",
        f"{'Mean Sortino':<22} {baseline.summary['mean_sortino']:>10.2f} {costs.summary['mean_sortino']:>10.2f} {full.summary['mean_sortino']:>10.2f}",
        f"{'Mean Calmar':<22} {baseline.summary['mean_calmar']:>10.2f} {costs.summary['mean_calmar']:>10.2f} {full.summary['mean_calmar']:>10.2f}",
        f"{'Mean Max DD':<22} {baseline.summary['mean_max_dd']*100:>9.1f}% {costs.summary['mean_max_dd']*100:>9.1f}% {full.summary['mean_max_dd']*100:>9.1f}%",
        f"{'Mean OOS Return':<22} {baseline.summary['mean_return']*100:>9.1f}% {costs.summary['mean_return']*100:>9.1f}% {full.summary['mean_return']*100:>9.1f}%",
        f"{'Mean Alpha vs URTH':<22} {b_alpha*100:>9.1f}% {c_alpha*100:>9.1f}% {f_alpha*100:>9.1f}%",
        f"{'Beat URTH rate':<22} {baseline.summary['beat_rate']:>9.0%}  {costs.summary['beat_rate']:>9.0%}  {full.summary['beat_rate']:>9.0%}",
        f"{'Win Rate':<22} {baseline.summary['mean_win_rate']:>9.0%}  {costs.summary['mean_win_rate']:>9.0%}  {full.summary['mean_win_rate']:>9.0%}",
        f"{'Total Trades':<22} {int(baseline.summary['total_trades']):>10} {int(costs.summary['total_trades']):>10} {int(full.summary['total_trades']):>10}",
        f"{'='*65}",
        "",
        f"Cost drag on alpha    : {cost_drag*100:+.2f}% ({cost_pct:.0f}% of baseline alpha)",
        f"DD reduction (corr)   : {dd_reduction*100:.1f}%",
        f"Trade reduction (corr): {trade_reduction:.0f}%",
    ]

    if cost_pct > 30:
        lines.append("")
        lines.append("⚠️  WARNING: Transaction costs consume >30% of the alpha.")
        lines.append("   This means the baseline Sharpe is substantially overstated.")
        lines.append("   The strategy needs higher alpha generation or fewer trades.")

    if f_alpha < 0:
        lines.append("")
        lines.append("❌  HONEST ASSESSMENT: After realistic costs + correlation filter,")
        lines.append("   the strategy produces NEGATIVE alpha vs URTH.")
        lines.append("   A passive URTH ETF would have outperformed — consider whether")
        lines.append("   the added complexity is justified.")
    elif f_alpha > 0.02:
        lines.append("")
        lines.append("✅  After realistic costs and filters, positive alpha remains.")
        lines.append("   This is a meaningful signal — the edge is not purely cost-free.")

    if dd_reduction > 0.15:
        lines.append(f"\n✅  Correlation filter reduced drawdown by {dd_reduction*100:.0f}% —")
        lines.append("   meaningful diversification benefit from the filter.")

    lines.append(f"{'='*65}")
    return "\n".join(lines)


# ── Prepared data cache (download once) ───────────────────────────────────────

@dataclass
class _PreparedData:
    prices: pd.DataFrame
    ohlcv: dict[str, pd.DataFrame]
    entries: pd.DataFrame
    exits: pd.DataFrame
    sl_fracs: pd.DataFrame
    slippage_baseline: pd.DataFrame
    slippage_costs: pd.DataFrame
    benchmark_prices: pd.Series
    windows: list[WFWindow]


# ── Main Orchestrator ──────────────────────────────────────────────────────────

class WalkForwardValidator:
    def __init__(self, cfg_path: Path = _CFG_PATH) -> None:
        self.cfg = load_config(cfg_path)
        self._prepared: _PreparedData | None = None

    def _prepare(self) -> _PreparedData:
        """Download data and compute signals once; cache for all 3 configs."""
        if self._prepared is not None:
            return self._prepared

        cfg = self.cfg
        data_cfg = cfg["data"]
        symbols: list[str] = flatten_watchlist(cfg)
        benchmark_sym: str = data_cfg["benchmark"]
        all_syms = list(dict.fromkeys(symbols + [benchmark_sym, "SPY"]))

        print("Downloading price data …")
        ohlcv = download_ohlcv(all_syms, data_cfg["start_date"], data_cfg["end_date"])
        prices = pd.DataFrame({sym: df["close"] for sym, df in ohlcv.items()}).sort_index()

        benchmark_prices = prices.pop(benchmark_sym) if benchmark_sym in prices.columns else pd.Series(dtype=float)
        spy_close = prices.pop("SPY") if "SPY" in prices.columns else pd.Series(dtype=float)

        traded_ohlcv = {sym: df for sym, df in ohlcv.items() if sym in symbols}

        print("Computing signals …")
        entries, exits, sl_fracs = compute_signals(traded_ohlcv, spy_close, cfg)
        slippage_baseline = build_baseline_slippage_df(traded_ohlcv, cfg)
        slippage_costs = build_cost_slippage_df(traded_ohlcv, cfg)

        # Align to common index
        common_idx = prices.index.intersection(entries.index)
        prices = prices.loc[common_idx]
        entries = entries.reindex(common_idx, fill_value=False)
        exits = exits.reindex(common_idx, fill_value=False)
        sl_fracs = sl_fracs.reindex(common_idx, fill_value=0.0)
        slippage_baseline = slippage_baseline.reindex(common_idx, fill_value=0.001)
        slippage_costs = slippage_costs.reindex(common_idx, fill_value=0.001)

        windows = generate_windows(cfg)

        self._prepared = _PreparedData(
            prices=prices,
            ohlcv=traded_ohlcv,
            entries=entries,
            exits=exits,
            sl_fracs=sl_fracs,
            slippage_baseline=slippage_baseline,
            slippage_costs=slippage_costs,
            benchmark_prices=benchmark_prices,
            windows=windows,
        )
        return self._prepared

    def run(self, backtest_cfg: BacktestConfig | None = None) -> WalkForwardResult:
        """Run all OOS windows for one BacktestConfig."""
        if backtest_cfg is None:
            backtest_cfg = BacktestConfig()

        data = self._prepare()
        slippage_df = data.slippage_costs if backtest_cfg.use_costs else data.slippage_baseline

        print(f"Running {len(data.windows)} OOS windows for config: {backtest_cfg.name}\n")

        results: list[WindowMetrics] = []
        all_filter_logs: list[str] = []

        for win in data.windows:
            print(
                f"  Window {win.window_id:02d} | OOS {win.oos_start} → {win.oos_end} … ",
                end="", flush=True,
            )
            metrics, flog = run_oos_backtest(
                window=win,
                prices=data.prices,
                entries=data.entries,
                exits=data.exits,
                sl_fracs=data.sl_fracs,
                slippage_df=slippage_df,
                benchmark_prices=data.benchmark_prices,
                cfg=self.cfg,
                backtest_cfg=backtest_cfg,
            )
            if metrics is None:
                print("skipped (no data)")
                continue

            all_filter_logs.extend(flog)
            results.append(metrics)
            print(
                f"ret={metrics.total_return*100:+.1f}% | "
                f"Sharpe={metrics.sharpe:.2f} | "
                f"alpha={metrics.alpha*100:+.1f}% | "
                f"trades={metrics.num_trades}"
            )

        if not results:
            raise RuntimeError("No OOS windows produced results — check data availability.")

        if all_filter_logs:
            print(f"\n  Filter log ({len(all_filter_logs)} events):")
            print("  " + summarise_filter_log(all_filter_logs).replace("\n", "\n  "))

        metrics_df = pd.DataFrame([r.__dict__ for r in results])
        summary = _aggregate(results)
        verdict = _verdict(summary, backtest_cfg.name)
        print(f"\n{verdict}\n")

        return WalkForwardResult(
            name=backtest_cfg.name,
            windows=results,
            metrics_df=metrics_df,
            summary=summary,
            verdict=verdict,
            filter_log=all_filter_logs,
        )

    def run_diagnostics(self, output_dir: Path | None = None) -> None:
        """Run pre-backtest signal diagnostics (funnel + frequency check)."""
        from backtest.signal_diagnostics import (
            run_all_funnels,
            check_signal_frequency,
            print_funnel_report,
            save_funnel_csv,
        )

        data = self._prepare()
        cfg = self.cfg

        spy_close = (
            data.ohlcv["SPY"]["close"]
            if "SPY" in data.ohlcv
            else pd.Series(dtype=float)
        )

        print("\n" + "=" * 65)
        print("PRE-BACKTEST SIGNAL DIAGNOSTICS")
        print("=" * 65)

        check_signal_frequency(
            data.entries,
            cfg["data"]["start_date"],
            cfg["data"]["end_date"],
        )

        funnel_df = run_all_funnels(
            data.ohlcv,
            spy_close,
            cfg,
            start=cfg["data"]["start_date"],
            end=cfg["data"]["end_date"],
        )
        print_funnel_report(funnel_df, cfg)

        out = output_dir or Path(cfg["report"]["output_path"]).parent
        out.mkdir(parents=True, exist_ok=True)
        csv_path = save_funnel_csv(funnel_df, out)
        print(f"  Funnel CSV → {csv_path.resolve()}\n")

    def run_comparison(self) -> ComparisonResult:
        """Run all 3 configs (data downloaded once) and return comparison."""
        self._prepare()  # load once

        print("\n" + "=" * 65)
        print("WALK-FORWARD COMPARISON: 3 Configurations")
        print("=" * 65)

        results = [self.run(bc) for bc in CONFIGS]
        comparison_verdict = _comparison_verdict(*results)
        print(comparison_verdict)

        return ComparisonResult(
            baseline=results[0],
            with_costs=results[1],
            with_costs_and_correlation=results[2],
        )
