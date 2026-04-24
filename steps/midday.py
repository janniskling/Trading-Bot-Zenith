"""Step 3 – Midday Check (16:30 UTC = 18:30 Berlin)"""
from __future__ import annotations

import sys
import traceback

from core.alpaca_client import AlpacaClient
from core.market_utils import should_skip_today, count_trading_days_since
from core.portfolio_manager import PortfolioManager
from core.telegram_bot import TelegramNotifier
from config.settings import MAX_HOLD_DAYS
from memory.log_manager import LogManager
from strategy.momentum_ema import MomentumEMAStrategy


def run_midday() -> None:
    client = AlpacaClient()
    telegram = TelegramNotifier()

    try:
        if should_skip_today(client):
            print("Market closed today. Skipping midday step.")
            sys.exit(0)

        log = LogManager()
        portfolio = PortfolioManager(client)
        strategy = MomentumEMAStrategy(client)
        state = portfolio.get_portfolio_state()

        positions = state.positions
        actions_taken: list[str] = []

        for pos in positions:
            if not portfolio.can_trade(state):
                print("Daily trade limit reached, stopping.")
                break

            # Time-in-trade filter: exit stagnant positions after MAX_HOLD_DAYS
            entry_date = client.get_position_entry_date(pos.symbol)
            if entry_date:
                days_held = count_trading_days_since(str(entry_date))
                if days_held >= MAX_HOLD_DAYS:
                    print(f"Closing {pos.symbol}: held {days_held} trading days (max {MAX_HOLD_DAYS})")
                    client.close_position(pos.symbol)
                    actions_taken.append(f"CLOSED {pos.symbol} – Haltedauer {days_held} Tage erreicht")
                    state = portfolio.get_portfolio_state()
                    continue

            exit_signal = strategy.check_exit_signal(pos.symbol)
            if exit_signal:
                print(f"Closing {pos.symbol}: {exit_signal.reason}")
                client.close_position(pos.symbol)
                actions_taken.append(f"CLOSED {pos.symbol} – {exit_signal.reason}")
                state = portfolio.get_portfolio_state()

        log_lines = [
            f"Positionen geprüft: {len(positions)}",
            f"Aktionen: {len(actions_taken)}",
        ]
        if actions_taken:
            log_lines.extend(actions_taken)
        log.append_to_daily_log("Midday Check", "\n".join(log_lines))

        telegram.send_midday_check(state.positions)
        if actions_taken:
            telegram.send_midday_actions(actions_taken)

        print(f"Midday complete. {len(actions_taken)} positions closed.")

    except Exception as e:
        print(f"Midday error: {e}\n{traceback.format_exc()}")
        try:
            telegram.send_error("midday", str(e))
        except Exception:
            pass
        sys.exit(1)
