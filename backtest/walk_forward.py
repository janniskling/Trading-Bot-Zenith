"""Walk-Forward Validation for the Zenith EMA-Momentum strategy.

Splits history into rolling IS/OOS windows, runs vectorbt backtests on each
OOS slice, and aggregates metrics across all windows.
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

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Config loading ─────────────────────────────────────────────────────────────

_CFG_PATH = Path(__file__).parent / "config.yaml"


def load_config(path: Path = _CFG_PATH) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Data download ──────────────────────────────────────────────────────────────

def download_prices(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted daily close prices for all symbols via yfinance.

    Returns a DataFrame with symbols as columns and DatetimeIndex as rows.
    Missing symbols are dropped with a warning.
    """
    raw = yf.download(
        symbols,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": symbols[0]})

    missing = [s for s in symbols if s not in prices.columns]
    if missing:
        print(f"[download] WARNING — symbols not found, skipping: {missing}")
    return prices.dropna(how="all")


def download_ohlcv(symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Download OHLCV for each symbol; returns {symbol: DataFrame}."""
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
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    atr_s = tr.ewm(com=period - 1, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=close.index).ewm(com=period - 1, min_periods=period).mean() / atr_s
    minus_di = 100 * pd.Series(minus_dm, index=close.index).ewm(com=period - 1, min_periods=period).mean() / atr_s

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.ewm(com=period - 1, min_periods=period).mean()


def compute_signals(
    ohlcv: dict[str, pd.DataFrame],
    spy_close: pd.Series,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute entry/exit signals and ATR-fraction stops for all symbols.

    Returns:
        entries  – boolean DataFrame (date × symbol), True on entry bar
        exits    – boolean DataFrame (date × symbol), True on exit bar
        sl_fracs – float DataFrame (date × symbol), stop-loss fraction below entry
    """
    sc = cfg["strategy"]
    ema_fast = sc["ema_fast"]
    ema_mid = sc["ema_mid"]
    ema_slow = sc["ema_slow"]
    rsi_period = sc["rsi_period"]
    rsi_min = sc["rsi_oversold"]
    rsi_max = sc["rsi_overbought"]
    adx_period = sc["adx_period"]
    adx_min = sc["adx_min"]
    atr_period = sc["atr_period"]
    atr_mult = sc["atr_stop_multiplier"]
    vol_factor = sc["volume_factor"]
    spy_ema_period = sc["spy_ema_trend"]

    # Bear-market mask: SPY above EMA-200
    spy_ema200 = _ema(spy_close, spy_ema_period)
    spy_bull = spy_close >= spy_ema200

    all_entries: dict[str, pd.Series] = {}
    all_exits: dict[str, pd.Series] = {}
    all_sl_fracs: dict[str, pd.Series] = {}

    for sym, df in ohlcv.items():
        close = df["close"]

        fast = _ema(close, ema_fast)
        mid = _ema(close, ema_mid)
        slow = _ema(close, ema_slow)

        rsi = _rsi(close, rsi_period)
        adx = _adx(df, adx_period)
        atr = _atr(df, atr_period)

        vol_ma = df["volume"].rolling(20).mean()
        vol_ok = df["volume"] > vol_factor * vol_ma

        # EMA crossover: fast > mid > slow (bullish stack)
        ema_bull = (fast > mid) & (mid > slow)
        # Crossover event: fast just crossed above mid
        crossed = (fast > mid) & (fast.shift(1) <= mid.shift(1))

        rsi_ok = (rsi > rsi_min) & (rsi < rsi_max)
        adx_ok = adx >= adx_min

        # Align bear mask to symbol index
        bull_aligned = spy_bull.reindex(close.index).fillna(False)

        entry = crossed & ema_bull & rsi_ok & adx_ok & vol_ok & bull_aligned

        # ATR stop as fraction below entry price (vectorbt sl_stop)
        atr_frac = (atr_mult * atr / close).clip(upper=0.15)  # cap at 15%

        # Exit: EMA stack flips bearish (fast < mid)
        exit_sig = fast < mid

        all_entries[sym] = entry.astype(bool)
        all_exits[sym] = exit_sig.astype(bool)
        all_sl_fracs[sym] = atr_frac

    symbols = list(all_entries.keys())
    entries = pd.DataFrame(all_entries)[symbols]
    exits = pd.DataFrame(all_exits)[symbols]
    sl_fracs = pd.DataFrame(all_sl_fracs)[symbols]

    return entries, exits, sl_fracs


# ── Walk-Forward Windows ───────────────────────────────────────────────────────

@dataclass
class WFWindow:
    is_start: date
    is_end: date
    oos_start: date
    oos_end: date
    window_id: int


def generate_windows(cfg: dict[str, Any]) -> list[WFWindow]:
    """Generate IS/OOS date pairs for the walk-forward walk."""
    wf = cfg["walk_forward"]
    data = cfg["data"]

    # Burn first 12 months for indicator warm-up; IS windows start after that
    universe_start = pd.Timestamp(data["start_date"])
    is_start = universe_start + relativedelta(months=12)
    total_end = pd.Timestamp(data["end_date"])

    is_months = wf["in_sample_months"]
    oos_months = wf["out_of_sample_months"]
    stride = wf["stride_months"]

    windows: list[WFWindow] = []
    wid = 1
    while True:
        is_end = is_start + relativedelta(months=is_months) - relativedelta(days=1)
        oos_start = is_end + relativedelta(days=1)
        oos_end = oos_start + relativedelta(months=oos_months) - relativedelta(days=1)

        if oos_end > total_end:
            break

        windows.append(
            WFWindow(
                is_start=is_start.date(),
                is_end=is_end.date(),
                oos_start=oos_start.date(),
                oos_end=oos_end.date(),
                window_id=wid,
            )
        )
        is_start = is_start + relativedelta(months=stride)
        wid += 1

    return windows


# ── Slippage ───────────────────────────────────────────────────────────────────

def build_slippage_df(
    ohlcv: dict[str, pd.DataFrame], cfg: dict[str, Any]
) -> pd.DataFrame:
    """Per-symbol, per-bar slippage fraction estimated from intraday range."""
    sc = cfg["slippage"]
    spread_factor = sc["spread_factor"]
    min_pct = sc["min_pct"]
    max_pct = sc["max_pct"]

    result: dict[str, pd.Series] = {}
    for sym, df in ohlcv.items():
        raw_spread = (df["high"] - df["low"]) / df["close"]
        slippage = (raw_spread * spread_factor).clip(lower=min_pct, upper=max_pct)
        result[sym] = slippage

    return pd.DataFrame(result)


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
    """Extract metrics from a vectorbt Portfolio object."""
    stats = pf.stats()

    total_return = _safe(stats.get("Total Return [%]", 0)) / 100
    sharpe = _safe(stats.get("Sharpe Ratio", 0))
    sortino = _safe(stats.get("Sortino Ratio", 0))
    calmar = _safe(stats.get("Calmar Ratio", 0))
    max_dd = _safe(stats.get("Max Drawdown [%]", 0)) / 100
    win_rate = _safe(stats.get("Win Rate [%]", 0)) / 100
    profit_factor = _safe(stats.get("Profit Factor", 0))
    num_trades = int(_safe(stats.get("Total Trades", 0)))

    # Benchmark return over same OOS period
    bm_slice = benchmark_oos.loc[
        str(window.oos_start): str(window.oos_end)
    ]
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


# ── Core Backtest per Window ───────────────────────────────────────────────────

def run_oos_backtest(
    window: WFWindow,
    prices: pd.DataFrame,
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    sl_fracs: pd.DataFrame,
    slippage_df: pd.DataFrame,
    benchmark_prices: pd.Series,
    cfg: dict[str, Any],
) -> WindowMetrics | None:
    """Run a single OOS backtest window and return its metrics."""
    oos_slice = slice(str(window.oos_start), str(window.oos_end))

    price_oos = prices.loc[oos_slice]
    entry_oos = entries.loc[oos_slice]
    exit_oos = exits.loc[oos_slice]
    sl_oos = sl_fracs.loc[oos_slice]
    slip_oos = slippage_df.loc[oos_slice]

    # Align columns
    cols = [c for c in price_oos.columns if c in entry_oos.columns]
    if not cols:
        return None

    price_oos = price_oos[cols]
    entry_oos = entry_oos[cols]
    exit_oos = exit_oos[cols]
    sl_oos = sl_oos[cols]
    slip_oos = slip_oos[cols]

    # Median slippage per symbol (scalar for vectorbt)
    avg_slip = slip_oos.median().fillna(cfg["slippage"]["min_pct"]).to_dict()

    rc = cfg["risk"]
    init_cash = 10_000.0  # normalised starting equity

    try:
        pf = vbt.Portfolio.from_signals(
            close=price_oos,
            entries=entry_oos,
            exits=exit_oos,
            sl_stop=sl_oos,
            init_cash=init_cash,
            size=rc["risk_per_trade_pct"],
            size_type="valuepercent",
            fees=0.0,
            slippage={col: avg_slip.get(col, 0.001) for col in cols},
            accumulate=False,
            freq="D",
        )
    except Exception as exc:
        print(f"  [window {window.window_id}] vectorbt error: {exc}")
        return None

    return compute_metrics(pf, window, benchmark_prices)


# ── Aggregation & Verdict ──────────────────────────────────────────────────────

@dataclass
class WalkForwardResult:
    windows: list[WindowMetrics]
    metrics_df: pd.DataFrame
    equity_curves: dict[int, pd.Series]  # window_id → equity curve
    summary: dict[str, float] = field(default_factory=dict)
    verdict: str = ""


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
    }


def _verdict(summary: dict[str, float]) -> str:
    beat = summary["beat_rate"]
    sharpe = summary["mean_sharpe"]
    alpha = summary["mean_alpha"]
    dd = summary["mean_max_dd"]

    lines = ["=" * 60, "WALK-FORWARD VERDICT", "=" * 60]

    if beat >= 0.65 and sharpe >= 0.5 and alpha > 0.02:
        lines.append("✅  STRATEGY HOLDS OUT-OF-SAMPLE")
        lines.append(f"    Beat benchmark in {beat:.0%} of windows")
        lines.append(f"    Mean Sharpe {sharpe:.2f} | Mean alpha {alpha*100:+.1f}%/window")
    elif beat >= 0.50:
        lines.append("⚠️   STRATEGY MARGINALLY VIABLE")
        lines.append(f"    Beat benchmark in {beat:.0%} of windows (50% = coin-flip)")
        lines.append("    Consider tightening filters before live trading.")
    else:
        lines.append("❌  STRATEGY UNDERPERFORMS BUY-AND-HOLD URTH")
        lines.append(f"    Beat benchmark in only {beat:.0%} of windows")
        lines.append(f"    Mean alpha {alpha*100:+.1f}% — negative edge detected.")
        lines.append("    Deploying this strategy live is NOT recommended without")
        lines.append("    significant parameter changes or additional alpha sources.")

    lines += [
        "",
        f"  Windows tested  : {summary['total_windows']:.0f}",
        f"  Positive return : {summary['positive_return_rate']:.0%} of windows",
        f"  Mean Sortino    : {summary['mean_sortino']:.2f}",
        f"  Mean Calmar     : {summary['mean_calmar']:.2f}",
        f"  Mean Max DD     : {summary['mean_max_dd']*100:.1f}%",
        f"  Mean Win Rate   : {summary['mean_win_rate']:.0%}",
        f"  Mean Profit Fct : {summary['mean_profit_factor']:.2f}",
        "=" * 60,
    ]
    return "\n".join(lines)


# ── Main Orchestrator ──────────────────────────────────────────────────────────

class WalkForwardValidator:
    """Top-level orchestrator for the walk-forward backtest."""

    def __init__(self, cfg_path: Path = _CFG_PATH) -> None:
        self.cfg = load_config(cfg_path)

    def run(self) -> WalkForwardResult:
        cfg = self.cfg
        data_cfg = cfg["data"]

        print("Downloading price data …")
        symbols: list[str] = data_cfg["symbols"]
        benchmark_sym: str = data_cfg["benchmark"]
        all_symbols = list(dict.fromkeys(symbols + [benchmark_sym, "SPY"]))

        ohlcv = download_ohlcv(all_symbols, data_cfg["start_date"], data_cfg["end_date"])
        prices = pd.DataFrame({sym: df["close"] for sym, df in ohlcv.items()}).sort_index()

        benchmark_prices = prices.pop(benchmark_sym) if benchmark_sym in prices.columns else pd.Series(dtype=float)
        spy_close = prices.pop("SPY") if "SPY" in prices.columns else pd.Series(dtype=float)

        # Keep only traded symbols
        traded_ohlcv = {sym: df for sym, df in ohlcv.items() if sym in symbols}

        print("Computing signals …")
        entries, exits, sl_fracs = compute_signals(traded_ohlcv, spy_close, cfg)
        slippage_df = build_slippage_df(traded_ohlcv, cfg)

        # Align everything to common index
        common_idx = prices.index.intersection(entries.index)
        prices = prices.loc[common_idx]
        entries = entries.reindex(common_idx, fill_value=False)
        exits = exits.reindex(common_idx, fill_value=False)
        sl_fracs = sl_fracs.reindex(common_idx, fill_value=0.0)
        slippage_df = slippage_df.reindex(common_idx, fill_value=0.001)

        windows = generate_windows(cfg)
        print(f"Running {len(windows)} OOS windows …\n")

        results: list[WindowMetrics] = []
        equity_curves: dict[int, pd.Series] = {}

        for win in windows:
            print(f"  Window {win.window_id:02d} | OOS {win.oos_start} → {win.oos_end} … ", end="", flush=True)
            metrics = run_oos_backtest(
                window=win,
                prices=prices,
                entries=entries,
                exits=exits,
                sl_fracs=sl_fracs,
                slippage_df=slippage_df,
                benchmark_prices=benchmark_prices,
                cfg=cfg,
            )
            if metrics is None:
                print("skipped (no data)")
                continue

            results.append(metrics)
            print(
                f"ret={metrics.total_return*100:+.1f}% | "
                f"Sharpe={metrics.sharpe:.2f} | "
                f"alpha={metrics.alpha*100:+.1f}% | "
                f"trades={metrics.num_trades}"
            )

        if not results:
            raise RuntimeError("No OOS windows produced valid results — check data availability.")

        metrics_df = pd.DataFrame([r.__dict__ for r in results])
        summary = _aggregate(results)
        verdict = _verdict(summary)

        print(f"\n{verdict}\n")

        return WalkForwardResult(
            windows=results,
            metrics_df=metrics_df,
            equity_curves=equity_curves,
            summary=summary,
            verdict=verdict,
        )
