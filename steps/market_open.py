"""Step 2 – Market Open (13:30 UTC = 15:30 Berlin)"""
from __future__ import annotations

import sys
import traceback

from core.alpaca_client import AlpacaClient
from core.market_utils import should_skip_today
from core.news_fetcher import fetch_earnings_calendar
from core.portfolio_manager import PortfolioManager
from core.telegram_bot import TelegramNotifier
from memory.log_manager import LogManager
from strategy.momentum_ema import MomentumEMAStrategy


def run_market_open() -> None:
    client = AlpacaClient()
    telegram = TelegramNotifier()

    try:
        if should_skip_today(client):
            print("Market closed today. Skipping market-open step.")
            sys.exit(0)

        # Wait for market to actually be open (GitHub Actions can be early)
        if not client.wait_for_market_open(max_wait_seconds=300):
            print("Market did not open within 5 minutes. Exiting.")
            sys.exit(0)

        log = LogManager()
        portfolio = PortfolioManager(client)
        strategy = MomentumEMAStrategy(client)

        # Parse candidates from today's pre-market log
        log_content = log.read_daily_log()
        candidate_symbols = log.parse_candidates_from_log(log_content)

        if not candidate_symbols:
            print("No candidates from pre-market log.")
            log.append_to_daily_log("Market Open", "Keine Kandidaten aus Pre-Market-Phase.")
            telegram.send_market_open_summary([])
            return

        state = portfolio.get_portfolio_state()
        print(f"Portfolio: {state.equity:.2f}€ | Positions: {len(state.positions)} | Trades today: {state.trade_count_today}")

        # Earnings check — avoid entries within 3 days of earnings
        earnings = fetch_earnings_calendar(candidate_symbols)
        earnings_soon = {e.symbol for e in earnings if e.days_away <= 3}

        orders_placed = []
        order_log_lines: list[str] = []

        for symbol in candidate_symbols:
            if not portfolio.can_trade(state):
                print("Trading limit reached, stopping.")
                break

            if symbol in state.open_symbols:
                print(f"Already in position: {symbol}")
                continue

            if symbol in earnings_soon:
                print(f"Skipping {symbol} — earnings in ≤3 days")
                order_log_lines.append(f"⚠️ {symbol}: Übersprungen (Earnings in ≤3 Tagen)")
                continue

            signal = strategy.check_entry_signal(symbol)
            if signal is None:
                print(f"No entry signal for {symbol}")
                continue

            entry = signal.entry_price
            stop = portfolio.get_stop_price(entry)
            take_profit = portfolio.get_take_profit_price(entry)
            qty = portfolio.calculate_position_size(entry, stop, state.equity)

            if qty <= 0:
                print(f"Position size 0 for {symbol}, skipping")
                continue

            print(f"Placing bracket order: {symbol} qty={qty} entry={entry:.2f} sl={stop:.2f} tp={take_profit:.2f}")
            order = client.submit_bracket_order(
                symbol=symbol, qty=qty,
                entry_price=entry, stop_loss=stop, take_profit=take_profit,
            )
            orders_placed.append(order)
            order_log_lines.append(
                f"BUY {qty}x **{symbol}** @ {entry:.2f} | SL: {stop:.2f} | TP: {take_profit:.2f} | Score: {signal.confidence_score}/100"
            )

            # Refresh state after each order (trade count matters)
            state = portfolio.get_portfolio_state()

        log.append_to_daily_log(
            "Market Open – Orders",
            "\n".join(order_log_lines) if order_log_lines else "Keine Orders platziert.",
        )
        telegram.send_market_open_summary(orders_placed)
        print(f"Market open complete. {len(orders_placed)} orders placed.")

    except Exception as e:
        print(f"Market-open error: {e}\n{traceback.format_exc()}")
        try:
            telegram.send_error("market_open", str(e))
        except Exception:
            pass
        sys.exit(1)
