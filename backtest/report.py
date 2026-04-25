"""PDF report generator for Walk-Forward Validation results."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from backtest.walk_forward import WalkForwardResult, load_config


_CFG_PATH = Path(__file__).parent / "config.yaml"
_PALETTE = {
    "green":  "#2ecc71",
    "red":    "#e74c3c",
    "blue":   "#3498db",
    "orange": "#e67e22",
    "grey":   "#95a5a6",
    "dark":   "#2c3e50",
    "light":  "#ecf0f1",
}


def _bar_color(values: pd.Series) -> list[str]:
    return [_PALETTE["green"] if v >= 0 else _PALETTE["red"] for v in values]


def _title_page(pdf: PdfPages, result: WalkForwardResult) -> None:
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor(_PALETTE["dark"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(_PALETTE["dark"])
    ax.axis("off")

    s = result.summary
    lines = [
        ("ZENITH – Walk-Forward Validation Report", 0.88, 22, "white"),
        (f"Windows tested: {s['total_windows']:.0f}", 0.75, 13, _PALETTE["light"]),
        (
            f"Beat URTH in {s['beat_rate']:.0%} of windows  |  "
            f"Mean alpha {s['mean_alpha']*100:+.1f}%/window",
            0.68, 13, _PALETTE["light"],
        ),
        (
            f"Mean Sharpe {s['mean_sharpe']:.2f}  |  "
            f"Mean Sortino {s['mean_sortino']:.2f}  |  "
            f"Mean Calmar {s['mean_calmar']:.2f}",
            0.61, 12, _PALETTE["light"],
        ),
        (
            f"Mean Max DD {s['mean_max_dd']*100:.1f}%  |  "
            f"Mean Win Rate {s['mean_win_rate']:.0%}  |  "
            f"Mean Profit Factor {s['mean_profit_factor']:.2f}",
            0.54, 12, _PALETTE["light"],
        ),
    ]
    for text, y, size, color in lines:
        ax.text(0.5, y, text, transform=ax.transAxes,
                ha="center", va="center", color=color, fontsize=size, fontweight="bold")

    # Verdict box
    verdict_short = result.verdict.split("\n")[2]  # emoji + status line
    verdict_color = _PALETTE["green"] if "✅" in verdict_short else (
        _PALETTE["orange"] if "⚠️" in verdict_short else _PALETTE["red"]
    )
    ax.text(0.5, 0.40, verdict_short, transform=ax.transAxes,
            ha="center", va="center", color=verdict_color, fontsize=16, fontweight="bold")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _returns_overview(pdf: PdfPages, df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("OOS Returns per Window", fontsize=14, fontweight="bold", color=_PALETTE["dark"])

    labels = [f"W{int(r['window_id'])}\n{str(r['oos_start'])[:7]}" for _, r in df.iterrows()]

    # Bar: strategy vs benchmark return
    x = np.arange(len(df))
    width = 0.35
    ax = axes[0]
    strat_colors = _bar_color(df["total_return"])
    bm_color = _PALETTE["blue"]
    ax.bar(x - width / 2, df["total_return"] * 100, width, color=strat_colors, label="Strategy", alpha=0.9)
    ax.bar(x + width / 2, df["benchmark_return"] * 100, width, color=bm_color, label="URTH Benchmark", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Return (%)")
    ax.set_title("Strategy vs Benchmark Return")
    ax.legend(fontsize=8)

    # Bar: alpha per window
    ax2 = axes[1]
    alpha_colors = _bar_color(df["alpha"])
    ax2.bar(x, df["alpha"] * 100, color=alpha_colors, alpha=0.9)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylabel("Alpha (%)")
    ax2.set_title("Alpha vs URTH per OOS Window")

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _risk_metrics(pdf: PdfPages, df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Risk Metrics per OOS Window", fontsize=14, fontweight="bold", color=_PALETTE["dark"])

    labels = [f"W{int(r['window_id'])}" for _, r in df.iterrows()]
    x = np.arange(len(df))

    panels = [
        (axes[0, 0], df["sharpe"],        "Sharpe Ratio",    0.5,   "above 0.5 is acceptable"),
        (axes[0, 1], df["sortino"],        "Sortino Ratio",   1.0,   "above 1.0 is good"),
        (axes[1, 0], df["max_drawdown"] * 100, "Max Drawdown (%)", -15, "worse than –15% is risky"),
        (axes[1, 1], df["win_rate"] * 100, "Win Rate (%)",    50,    "above 50% is positive"),
    ]

    for ax, values, title, threshold, note in panels:
        colors = [_PALETTE["green"] if v >= threshold else _PALETTE["red"] for v in values]
        ax.bar(x, values, color=colors, alpha=0.9)
        ax.axhline(threshold, color=_PALETTE["orange"], linewidth=1.2, linestyle="--", label=f"Threshold: {threshold}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f"{title}\n({note})", fontsize=9)
        ax.legend(fontsize=7)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _metrics_table(pdf: PdfPages, df: pd.DataFrame) -> None:
    display_cols = [
        "window_id", "oos_start", "oos_end",
        "total_return", "sharpe", "sortino", "calmar",
        "max_drawdown", "win_rate", "profit_factor",
        "num_trades", "alpha",
    ]
    table_df = df[display_cols].copy()
    pct_cols = ["total_return", "max_drawdown", "win_rate", "alpha"]
    for col in pct_cols:
        table_df[col] = (table_df[col] * 100).round(1).astype(str) + "%"
    for col in ["sharpe", "sortino", "calmar", "profit_factor"]:
        table_df[col] = table_df[col].round(2)

    table_df.columns = [
        "#", "OOS Start", "OOS End",
        "Return", "Sharpe", "Sortino", "Calmar",
        "Max DD", "Win%", "PF",
        "Trades", "Alpha",
    ]

    fig, ax = plt.subplots(figsize=(14, max(4, len(df) * 0.45 + 2)))
    ax.axis("off")
    tbl = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor(_PALETTE["dark"])
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f2f2f2")

    fig.suptitle("Full OOS Metrics Table", fontsize=13, fontweight="bold", color=_PALETTE["dark"])
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _verdict_page(pdf: PdfPages, result: WalkForwardResult) -> None:
    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor(_PALETTE["light"])
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.set_facecolor(_PALETTE["light"])
    ax.axis("off")

    verdict_text = result.verdict
    ax.text(
        0.5, 0.5, verdict_text,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=10,
        fontfamily="monospace",
        color=_PALETTE["dark"],
        wrap=True,
        bbox=dict(boxstyle="round,pad=1", facecolor="white", edgecolor=_PALETTE["grey"], linewidth=1.5),
    )
    fig.suptitle("Honest Assessment", fontsize=14, fontweight="bold", color=_PALETTE["dark"])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def generate_pdf(result: WalkForwardResult, cfg_path: Path = _CFG_PATH) -> Path:
    """Generate a multi-page PDF report and return its path."""
    cfg = load_config(cfg_path)
    output_path = Path(cfg["report"]["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = result.metrics_df

    with PdfPages(output_path) as pdf:
        _title_page(pdf, result)
        _returns_overview(pdf, df)
        _risk_metrics(pdf, df)
        _metrics_table(pdf, df)
        _verdict_page(pdf, result)

        # PDF metadata
        d = pdf.infodict()
        d["Title"] = "Zenith Walk-Forward Validation Report"
        d["Author"] = "Zenith Trading Bot"
        d["Subject"] = "Out-of-sample strategy validation"

    print(f"Report saved → {output_path.resolve()}")
    return output_path
