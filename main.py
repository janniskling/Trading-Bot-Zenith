"""Zenith Trading Bot – CLI entry point.

Usage:
    python main.py --step premarket
    python main.py --step market_open
    python main.py --step midday
    python main.py --step market_close
    python main.py --step nightly_reflection
    python main.py --step account_info          # quick connection test
"""
import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Zenith Trading Bot")
    parser.add_argument(
        "--step",
        required=True,
        choices=[
            "premarket",
            "market_open",
            "midday",
            "market_close",
            "nightly_reflection",
            "account_info",
        ],
        help="Which daily step to run",
    )
    args = parser.parse_args()

    if args.step == "account_info":
        _run_account_info()
        return

    step_map = {
        "premarket": "steps.premarket:run_premarket",
        "market_open": "steps.market_open:run_market_open",
        "midday": "steps.midday:run_midday",
        "market_close": "steps.market_close:run_market_close",
        "nightly_reflection": "steps.nightly_reflection:run_nightly_reflection",
    }

    module_path, func_name = step_map[args.step].split(":")
    import importlib
    module = importlib.import_module(module_path)
    func = getattr(module, func_name)
    func()


def _run_account_info() -> None:
    """Quick test: connects to Alpaca and prints account balance."""
    from core.alpaca_client import AlpacaClient
    from core.market_utils import is_market_open, format_german_time
    from datetime import datetime, timezone

    client = AlpacaClient()
    account = client.get_account()
    clock = client.get_clock()

    print("\n=== Zenith Account Info ===")
    print(f"Portfolio Value:  {account['equity']:,.2f} €")
    print(f"Cash:             {account['cash']:,.2f} €")
    print(f"Buying Power:     {account['buying_power']:,.2f} €")
    print(f"Day Trade Count:  {account['day_trade_count']}")
    print(f"Market Open:      {clock['is_open']}")
    if not clock["is_open"]:
        next_open = clock["next_open"]
        if hasattr(next_open, "astimezone"):
            print(f"Next Open:        {format_german_time(next_open)}")
    print()

    positions = client.get_positions()
    if positions:
        print(f"Open Positions ({len(positions)}):")
        for p in positions:
            print(
                f"  {p.symbol:6} {p.qty:6.0f} Stk @ {p.avg_entry_price:8.2f} | "
                f"P&L: {p.unrealized_pl:+8.2f} ({p.unrealized_plpc * 100:+.1f}%)"
            )
    else:
        print("No open positions.")
    print()


if __name__ == "__main__":
    main()
