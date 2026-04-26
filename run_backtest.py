#!/usr/bin/env python3
"""CLI entry point for the Zenith Walk-Forward Validation.

Usage:
    python run_backtest.py                      # single baseline run + PDF
    python run_backtest.py --compare            # 3-config comparison
    python run_backtest.py --no-pdf             # skip PDF, only stdout + CSV
    python run_backtest.py --config path/to/config.yaml
    python run_backtest.py --allow-curve-fitting  # override hard stop (use sparingly)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Zenith Walk-Forward Backtest")
    parser.add_argument("--config", type=Path, default=Path("backtest/config.yaml"),
                        help="Path to config.yaml")
    parser.add_argument("--compare", action="store_true",
                        help="Run all 3 configs (Baseline / +Costs / +Costs+Corr)")
    parser.add_argument("--no-pdf", action="store_true",
                        help="Skip PDF report generation")
    parser.add_argument("--allow-curve-fitting", action="store_true",
                        help="Override the parameter-change hard stop (document why!)")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"ERROR: config not found at {args.config}", file=sys.stderr)
        sys.exit(1)

    from backtest.integrity_check import check_integrity, print_expectation_check
    from backtest.walk_forward import WalkForwardValidator, load_config
    from backtest.report import generate_pdf, generate_comparison_pdf

    # ── Integrity gate (may raise SystemExit if hard stop triggered) ───────────
    check_integrity(allow_curve_fitting=args.allow_curve_fitting)

    cfg = load_config(args.config)
    output_dir = Path(cfg["report"]["output_path"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    validator = WalkForwardValidator(cfg_path=args.config)
    validator.run_diagnostics(output_dir=output_dir)

    if args.compare:
        comparison = validator.run_comparison()

        # Save per-config CSVs and filter logs
        for result in comparison.configs():
            csv_name = result.name.replace(" ", "_").replace("+", "plus") + "_metrics.csv"
            csv_path = output_dir / csv_name
            result.metrics_df.to_csv(csv_path, index=False)
            print(f"Metrics CSV → {csv_path.resolve()}")

            if result.filter_log:
                log_name = result.name.replace(" ", "_").replace("+", "plus") + "_filter_log.txt"
                log_path = output_dir / log_name
                log_path.write_text("\n".join(result.filter_log))
                print(f"Filter log → {log_path.resolve()}")

        # Expectation check uses baseline summary (no cost/filter adjustments)
        baseline = comparison.baseline.summary
        print_expectation_check(
            total_trades=int(baseline.get("total_trades", 0)),
            mean_sharpe=baseline.get("mean_sharpe", 0.0),
            mean_alpha=baseline.get("mean_alpha", 0.0),
        )

        if not args.no_pdf:
            generate_comparison_pdf(comparison, cfg_path=args.config)
    else:
        result = validator.run()

        csv_path = output_dir / "metrics.csv"
        result.metrics_df.to_csv(csv_path, index=False)
        print(f"Metrics CSV → {csv_path.resolve()}")

        print_expectation_check(
            total_trades=int(result.summary.get("total_trades", 0)),
            mean_sharpe=result.summary.get("mean_sharpe", 0.0),
            mean_alpha=result.summary.get("mean_alpha", 0.0),
        )

        if not args.no_pdf:
            generate_pdf(result, cfg_path=args.config)


if __name__ == "__main__":
    main()
