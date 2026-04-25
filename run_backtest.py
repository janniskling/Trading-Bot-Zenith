#!/usr/bin/env python3
"""CLI entry point for the Zenith Walk-Forward Validation.

Usage:
    python run_backtest.py [--config path/to/config.yaml] [--no-pdf]

Outputs:
    - Live progress to stdout
    - PDF report at the path configured in config.yaml
    - CSV of per-window metrics next to the PDF
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Zenith Walk-Forward Backtest")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("backtest/config.yaml"),
        help="Path to config.yaml (default: backtest/config.yaml)",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip PDF report generation",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"ERROR: config not found at {args.config}", file=sys.stderr)
        sys.exit(1)

    from backtest.walk_forward import WalkForwardValidator
    from backtest.report import generate_pdf

    validator = WalkForwardValidator(cfg_path=args.config)
    result = validator.run()

    # Save CSV of per-window metrics alongside the PDF
    from backtest.walk_forward import load_config
    cfg = load_config(args.config)
    output_dir = Path(cfg["report"]["output_path"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics.csv"
    result.metrics_df.to_csv(csv_path, index=False)
    print(f"Metrics CSV → {csv_path.resolve()}")

    if not args.no_pdf:
        generate_pdf(result, cfg_path=args.config)


if __name__ == "__main__":
    main()
