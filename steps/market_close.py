"""Step 4 – Market Close (20:00 UTC = 22:00 Berlin)"""
from __future__ import annotations

import sys
import traceback

import yfinance as yf

from core.alpaca_client import AlpacaClient
from core.market_utils import should_skip_today
from core.portfolio_manager import PortfolioManager
from core.telegram_bot import TelegramNotifier
from memory.log_manager import LogManager


def _fetch_benchmark_return() -> float | None:
    try:
        hist = yf.Ticker("URTH").history(period="2d")
        if len(hist) >= 2:
            return float(hist["Close"].iloc[-1] / hist["Close"].iloc[-2] - 1)
        return None
    except Exception:
        return None


def run_market_close() -> None:
    client = AlpacaClient()
    telegram = TelegramNotifier()

    try:
        if should_skip_today(client):
            print("Market closed today. Skipping market-close step.")
            sys.exit(0)

        log = LogManager()
        portfolio = PortfolioManager(client)

        # Cancel any remaining unfilled day orders
        client.cancel_all_orders()
        print("Cancelled open orders.")

        # Final portfolio state
        state = portfolio.get_portfolio_state()
        pnl = portfolio.calculate_daily_pnl()
        closed_trades = portfolio.get_closed_trades_today()
        trade_count = state.trade_count_today

        cash_pct = state.cash / state.equity * 100 if state.equity else 0

        benchmark_return = _fetch_benchmark_return()
        portfolio_return = state.day_pnl_pct / 100

        if benchmark_return is not None:
            alpha = portfolio_return - benchmark_return
            benchmark_line = (
                f"**Benchmark (URTH/MSCI World):** {benchmark_return * 100:+.2f}% | "
                f"**Alpha heute:** {alpha * 100:+.2f}%"
            )
        else:
            benchmark_line = "**Benchmark (URTH):** nicht verfügbar"

        eod_lines = [
            f"**Portfolio:** {state.equity:,.2f}€ ({state.day_pnl_pct:+.2f}%)",
            benchmark_line,
            f"**Cash:** {state.cash:,.2f}€ ({cash_pct:.1f}%)",
            f"**Offene Positionen:** {len(state.positions)}",
            f"**Trades heute:** {trade_count}/5",
            f"**Unrealisierter P&L:** {pnl.get('total', 0):+.2f}€",
        ]

        if state.positions:
            eod_lines.append("\n**Positionen:**")
            for p in state.positions:
                eod_lines.append(
                    f"  - {p.symbol}: {p.unrealized_pl:+.2f}€ ({p.unrealized_plpc * 100:+.1f}%) | "
                    f"{p.qty:.0f} Stk @ {p.avg_entry_price:.2f}"
                )

        log.append_to_daily_log("Market Close – Tagesabschluss", "\n".join(eod_lines))

        # Archive daily log to logs/YYYY-MM-DD.md
        log.archive_daily_log()

        # Send main daily summary via Telegram
        portfolio_dict = {
            "equity": state.equity,
            "cash": state.cash,
            "day_pnl": state.day_pnl,
            "day_pnl_pct": state.day_pnl_pct,
        }
        telegram.send_daily_summary(portfolio_dict, closed_trades, state.positions, trade_count)

        print(f"Market close complete. Portfolio: {state.equity:,.2f}€ ({state.day_pnl_pct:+.2f}%)")

    except Exception as e:
        print(f"Market-close error: {e}\n{traceback.format_exc()}")
        try:
            telegram.send_error("market_close", str(e))
        except Exception:
            pass
        sys.exit(1)
