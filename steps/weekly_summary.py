"""Step 6 – Weekly Summary (Sunday 18:00 UTC = 20:00 Berlin)"""
from __future__ import annotations

import sys
import traceback
from datetime import date, timedelta
from pathlib import Path

import yfinance as yf

from core.alpaca_client import AlpacaClient
from core.portfolio_manager import PortfolioManager
from core.telegram_bot import TelegramNotifier
from config.settings import LOGS_DIR


def _fetch_weekly_benchmark_return() -> float | None:
    try:
        hist = yf.Ticker("URTH").history(period="7d")
        if len(hist) >= 2:
            return float(hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1)
        return None
    except Exception:
        return None


def _read_week_logs() -> list[tuple[str, str]]:
    """Return (date_str, content) for the last 5 trading-day logs."""
    today = date.today()
    result: list[tuple[str, str]] = []
    for offset in range(7):
        day = today - timedelta(days=offset + 1)
        log_path = LOGS_DIR / f"{day.isoformat()}.md"
        if log_path.exists():
            result.append((day.isoformat(), log_path.read_text(encoding="utf-8")))
        if len(result) >= 5:
            break
    return list(reversed(result))


def _summarise_week(logs: list[tuple[str, str]]) -> str:
    if not logs:
        return "Keine Tages-Logs für diese Woche gefunden."
    days = [d for d, _ in logs]
    return f"Logs analysiert: {', '.join(days)}"


def run_weekly_summary() -> None:
    client = AlpacaClient()
    telegram = TelegramNotifier()

    try:
        portfolio = PortfolioManager(client)
        state = portfolio.get_portfolio_state()

        benchmark_return = _fetch_weekly_benchmark_return()
        logs = _read_week_logs()

        cash_pct = state.cash / state.equity * 100 if state.equity else 0
        open_count = len(state.positions)

        lines = [
            "📊 *Zenith – Wöchentlicher Report*",
            "",
            f"*Portfolio:* {state.equity:,.2f}€",
            f"*Cash:* {state.cash:,.2f}€ ({cash_pct:.1f}%)",
            f"*Offene Positionen:* {open_count}",
        ]

        if state.positions:
            lines.append("")
            lines.append("*Aktuelle Positionen:*")
            for p in state.positions:
                lines.append(
                    f"  • {p.symbol}: {p.unrealized_pl:+.2f}€ ({p.unrealized_plpc * 100:+.1f}%)"
                )

        if benchmark_return is not None:
            lines.append("")
            lines.append(f"*URTH (Woche):* {benchmark_return * 100:+.2f}%")

        log_summary = _summarise_week(logs)
        lines.append("")
        lines.append(f"*Tages-Logs:* {log_summary}")

        if not logs:
            lines.append("")
            lines.append("_Keine Trades diese Woche — Bot läuft im Beobachtungsmodus._")

        lines.append("")
        lines.append(f"_Stand: {date.today().isoformat()}_")

        telegram.send_message("\n".join(lines))
        print(f"Weekly summary sent. Portfolio: {state.equity:,.2f}€, {open_count} open positions.")

    except Exception as e:
        print(f"Weekly-summary error: {e}\n{traceback.format_exc()}")
        try:
            telegram.send_error("weekly_summary", str(e))
        except Exception:
            pass
        sys.exit(1)
