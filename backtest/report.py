"""PDF report generator for Walk-Forward Validation results."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from backtest.walk_forward import WalkForwardResult, ComparisonResult, load_config


_CFG_PATH = Path(__file__).parent / "config.yaml"

_COLORS = {
    "baseline":    "#3498db",   # blue
    "costs":       "#e67e22",   # orange
    "full":        "#2ecc71",   # green
    "benchmark":   "#95a5a6",   # grey
    "red":         "#e74c3c",
    "dark":        "#2c3e50",
    "light":       "#ecf0f1",
}


def _bar_color(values: pd.Series, threshold: float = 0.0) -> list[str]:
    return [_COLORS["full"] if v >= threshold else _COLORS["red"] for v in values]


# ── Single-run pages ───────────────────────────────────────────────────────────

def _title_page(pdf: PdfPages, result: WalkForwardResult) -> None:
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor(_COLORS["dark"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(_COLORS["dark"])
    ax.axis("off")

    s = result.summary
    lines = [
        (f"ZENITH — Walk-Forward Report: {result.name}", 0.88, 20, "white"),
        (f"Windows tested: {s['total_windows']:.0f}  |  Beat URTH in {s['beat_rate']:.0%} of windows", 0.74, 13, _COLORS["light"]),
        (f"Mean alpha {s['mean_alpha']*100:+.1f}%/window  |  Mean Sharpe {s['mean_sharpe']:.2f}  |  Mean Sortino {s['mean_sortino']:.2f}", 0.66, 12, _COLORS["light"]),
        (f"Mean Max DD {s['mean_max_dd']*100:.1f}%  |  Win Rate {s['mean_win_rate']:.0%}  |  Profit Factor {s['mean_profit_factor']:.2f}", 0.58, 12, _COLORS["light"]),
    ]
    for text, y, size, color in lines:
        ax.text(0.5, y, text, transform=ax.transAxes, ha="center", va="center",
                color=color, fontsize=size, fontweight="bold")

    verdict_line = result.verdict.split("\n")[2]
    vcolor = _COLORS["full"] if "✅" in verdict_line else (
        _COLORS["costs"] if "⚠️" in verdict_line else _COLORS["red"])
    ax.text(0.5, 0.42, verdict_line, transform=ax.transAxes, ha="center", va="center",
            color=vcolor, fontsize=15, fontweight="bold")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _returns_overview(pdf: PdfPages, df: pd.DataFrame, name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"OOS Returns per Window — {name}", fontsize=13, fontweight="bold")

    labels = [f"W{int(r['window_id'])}\n{str(r['oos_start'])[:7]}" for _, r in df.iterrows()]
    x = np.arange(len(df))
    w = 0.35

    ax = axes[0]
    ax.bar(x - w/2, df["total_return"]*100, w, color=_bar_color(df["total_return"]), label="Strategy", alpha=0.9)
    ax.bar(x + w/2, df["benchmark_return"]*100, w, color=_COLORS["benchmark"], label="URTH", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Return (%)"); ax.set_title("Strategy vs URTH Return"); ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.bar(x, df["alpha"]*100, color=_bar_color(df["alpha"]), alpha=0.9)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylabel("Alpha (%)"); ax2.set_title("Alpha vs URTH per Window")

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _metrics_table(pdf: PdfPages, df: pd.DataFrame, name: str) -> None:
    display_cols = ["window_id","oos_start","oos_end","total_return","sharpe","sortino",
                    "calmar","max_drawdown","win_rate","profit_factor","num_trades","alpha"]
    tdf = df[display_cols].copy()
    for col in ["total_return","max_drawdown","win_rate","alpha"]:
        tdf[col] = (tdf[col]*100).round(1).astype(str) + "%"
    for col in ["sharpe","sortino","calmar","profit_factor"]:
        tdf[col] = tdf[col].round(2)
    tdf.columns = ["#","OOS Start","OOS End","Return","Sharpe","Sortino","Calmar",
                   "Max DD","Win%","PF","Trades","Alpha"]

    fig, ax = plt.subplots(figsize=(14, max(4, len(df)*0.45+2)))
    ax.axis("off")
    tbl = ax.table(cellText=tdf.values, colLabels=tdf.columns, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.4)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor(_COLORS["dark"]); cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f5f5f5")

    fig.suptitle(f"Full OOS Metrics — {name}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Comparison pages ───────────────────────────────────────────────────────────

def _comparison_equity_curves(pdf: PdfPages, comparison: ComparisonResult) -> None:
    """Equity curves for all 3 configs + URTH benchmark on one plot."""
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle("Compounded OOS Equity Curves — 3 Configurations vs URTH",
                 fontsize=13, fontweight="bold", color=_COLORS["dark"])

    cfg_items = [
        (comparison.baseline,                 _COLORS["baseline"], "Baseline (no costs)"),
        (comparison.with_costs,               _COLORS["costs"],    "+Costs"),
        (comparison.with_costs_and_correlation, _COLORS["full"],   "+Costs+Correlation"),
    ]
    for result, color, label in cfg_items:
        curve = result.equity_curve()
        ax.plot(curve.index, curve.values, color=color, linewidth=2, label=label)

    bm_curve = comparison.baseline.benchmark_curve()
    ax.plot(bm_curve.index, bm_curve.values, color=_COLORS["benchmark"],
            linewidth=1.5, linestyle="--", label="URTH Benchmark")

    ax.axhline(1.0, color="black", linewidth=0.7, linestyle=":")
    ax.set_ylabel("Portfolio Value (start = 1.0)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _comparison_metrics_bars(pdf: PdfPages, comparison: ComparisonResult) -> None:
    """Side-by-side bar charts for Sharpe, Alpha, Max DD, Win Rate."""
    results = comparison.configs()
    names = [r.name for r in results]
    x = np.arange(len(results))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Risk/Return Metrics — 3 Configurations", fontsize=13,
                 fontweight="bold", color=_COLORS["dark"])

    panels = [
        (axes[0, 0], [r.summary["mean_sharpe"] for r in results],    "Mean Sharpe Ratio",       0.5),
        (axes[0, 1], [r.summary["mean_alpha"]*100 for r in results],  "Mean Alpha vs URTH (%)",  0.0),
        (axes[1, 0], [r.summary["mean_max_dd"]*100 for r in results], "Mean Max Drawdown (%)",   -15),
        (axes[1, 1], [r.summary["mean_win_rate"]*100 for r in results],"Win Rate (%)",            50),
    ]

    for ax, values, title, threshold in panels:
        bar_colors = [_COLORS["full"] if v >= threshold else _COLORS["red"] for v in values]
        ax.bar(x, values, color=bar_colors, alpha=0.85)
        ax.axhline(threshold, color=_COLORS["costs"], linewidth=1.2, linestyle="--",
                   label=f"Threshold: {threshold}")
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
        ax.set_title(title, fontsize=10); ax.legend(fontsize=7)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _comparison_table(pdf: PdfPages, comparison: ComparisonResult) -> None:
    results = comparison.configs()
    rows = [
        ["Mean Sharpe"] + [f"{r.summary['mean_sharpe']:.2f}" for r in results],
        ["Mean Sortino"] + [f"{r.summary['mean_sortino']:.2f}" for r in results],
        ["Mean Calmar"] + [f"{r.summary['mean_calmar']:.2f}" for r in results],
        ["Mean Max DD"] + [f"{r.summary['mean_max_dd']*100:.1f}%" for r in results],
        ["Mean OOS Return"] + [f"{r.summary['mean_return']*100:.1f}%" for r in results],
        ["Mean Alpha"] + [f"{r.summary['mean_alpha']*100:.1f}%" for r in results],
        ["Beat URTH Rate"] + [f"{r.summary['beat_rate']:.0%}" for r in results],
        ["Win Rate"] + [f"{r.summary['mean_win_rate']:.0%}" for r in results],
        ["Profit Factor"] + [f"{r.summary['mean_profit_factor']:.2f}" for r in results],
        ["Total Trades"] + [f"{int(r.summary['total_trades'])}" for r in results],
    ]
    col_labels = ["Metric"] + [r.name for r in results]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.8)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor(_COLORS["dark"]); cell.set_text_props(color="white", fontweight="bold")
        elif col == 0:
            cell.set_facecolor("#e8e8e8"); cell.set_text_props(fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f9f9f9")

    fig.suptitle("Summary Comparison Table", fontsize=13, fontweight="bold")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _verdict_page(pdf: PdfPages, verdict_text: str, title: str = "Honest Assessment") -> None:
    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor(_COLORS["light"])
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.set_facecolor(_COLORS["light"]); ax.axis("off")
    ax.text(0.5, 0.5, verdict_text, transform=ax.transAxes, ha="center", va="center",
            fontsize=9, fontfamily="monospace", color=_COLORS["dark"],
            bbox=dict(boxstyle="round,pad=1", facecolor="white", edgecolor="#aaa", linewidth=1.5))
    fig.suptitle(title, fontsize=13, fontweight="bold", color=_COLORS["dark"])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_pdf(result: WalkForwardResult, cfg_path: Path = _CFG_PATH) -> Path:
    """Single-run PDF report."""
    cfg = load_config(cfg_path)
    output_path = Path(cfg["report"]["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        _title_page(pdf, result)
        _returns_overview(pdf, result.metrics_df, result.name)
        _metrics_table(pdf, result.metrics_df, result.name)
        _verdict_page(pdf, result.verdict)
        d = pdf.infodict()
        d["Title"] = f"Zenith WFV Report — {result.name}"

    print(f"Report saved → {output_path.resolve()}")
    return output_path


def generate_comparison_pdf(
    comparison: ComparisonResult,
    cfg_path: Path = _CFG_PATH,
) -> Path:
    """3-configuration comparison PDF report."""
    cfg = load_config(cfg_path)
    output_path = Path(cfg["report"]["comparison_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from backtest.walk_forward import _comparison_verdict
    comp_verdict = _comparison_verdict(
        comparison.baseline,
        comparison.with_costs,
        comparison.with_costs_and_correlation,
    )

    with PdfPages(output_path) as pdf:
        # Comparison overview
        _comparison_equity_curves(pdf, comparison)
        _comparison_metrics_bars(pdf, comparison)
        _comparison_table(pdf, comparison)
        _verdict_page(pdf, comp_verdict, "3-Config Honest Assessment")

        # Individual per-config metric tables
        for result in comparison.configs():
            _returns_overview(pdf, result.metrics_df, result.name)
            _metrics_table(pdf, result.metrics_df, result.name)
            _verdict_page(pdf, result.verdict, f"Verdict — {result.name}")

        d = pdf.infodict()
        d["Title"] = "Zenith WFV — 3-Configuration Comparison"

    print(f"Comparison report saved → {output_path.resolve()}")
    return output_path
