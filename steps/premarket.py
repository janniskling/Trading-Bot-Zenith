"""Step 1 – Pre-Market Research (11:00 UTC = 13:00 Berlin)"""
from __future__ import annotations

import sys
import traceback

from core.alpaca_client import AlpacaClient
from core.llm_analyst import LLMAnalyst
from core.market_utils import should_skip_today, today_iso
from core.news_fetcher import (
    derive_market_mood,
    fetch_earnings_calendar,
    fetch_market_movers,
    fetch_rss_headlines,
    format_news_for_prompt,
)
from core.portfolio_manager import PortfolioManager
from core.telegram_bot import TelegramNotifier
from memory.log_manager import LogManager
from strategy.momentum_ema import MomentumEMAStrategy


def run_premarket() -> None:
    client = AlpacaClient()
    telegram = TelegramNotifier()

    try:
        if should_skip_today(client):
            print("Market closed today. Skipping pre-market step.")
            sys.exit(0)

        log = LogManager()
        portfolio = PortfolioManager(client)
        strategy = MomentumEMAStrategy(client)
        analyst = LLMAnalyst()

        # Init fresh daily log
        log.init_daily_log(today_iso())

        # Read memory
        learnings = log.read_learnings()
        watchlist_text, symbols = log.read_watchlist()

        if not symbols:
            print("Watchlist is empty. Add symbols to knowledge/watchlist.md")
            log.append_to_daily_log("Pre-Market Research", "Watchlist ist leer.")
            return

        # Fetch news
        headlines = fetch_rss_headlines(symbols)
        earnings = fetch_earnings_calendar(symbols)
        movers = fetch_market_movers()
        market_mood = derive_market_mood(movers)
        news_text = format_news_for_prompt(headlines, earnings)

        # Technical scoring
        print(f"Scoring {len(symbols)} symbols...")
        scored = strategy.score_candidates(symbols)

        # Portfolio state
        state = portfolio.get_portfolio_state()
        underdeployed = portfolio.is_underdeployed(state)

        # LLM analysis
        candidates = analyst.analyze_trade_candidates(
            watchlist_content=watchlist_text,
            learnings_content=learnings,
            news_summary=news_text,
            scored_candidates=scored,
            is_underdeployed=underdeployed,
        )

        # Log results
        movers_text = f"SPY: {movers.get('spy_change_pct', 0):+.1f}% | QQQ: {movers.get('qqq_change_pct', 0):+.1f}%"
        underdeploy_note = "\n⚡ **Kapitaldeployment-Priorität aktiv** – Cash > 20%" if underdeployed else ""
        log_content = (
            f"**Marktstimmung:** {market_mood}\n"
            f"**Futures:** {movers_text}{underdeploy_note}\n\n"
            f"**News-Zusammenfassung:**\n{news_text}\n\n"
            f"**Top-Kandidaten:**\n{analyst.format_candidates_for_log(candidates)}"
        )
        log.append_to_daily_log("Pre-Market Research", log_content)

        # Telegram
        telegram.send_premarket_brief(
            [{"symbol": c.symbol, "score": c.score, "reason": c.reason} for c in candidates],
            market_mood,
        )

        print(f"Pre-market complete. {len(candidates)} candidates identified.")

    except Exception as e:
        err = traceback.format_exc()
        print(f"Pre-market error: {e}")
        try:
            telegram.send_error("premarket", str(e))
        except Exception:
            pass
        sys.exit(1)
