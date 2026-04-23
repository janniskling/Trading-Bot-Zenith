"""Step 5 – Nightly Reflection (21:00 UTC = 23:00 Berlin)"""
from __future__ import annotations

import sys
import traceback

from core.llm_analyst import LLMAnalyst
from core.telegram_bot import TelegramNotifier
from memory.log_manager import LogManager


def run_nightly_reflection() -> None:
    telegram = TelegramNotifier()

    try:
        log = LogManager()
        analyst = LLMAnalyst()

        # Read today's archived log and existing learnings
        daily_log = log.read_daily_log()
        learnings = log.read_learnings()

        if not daily_log.strip() or daily_log.strip() == f"# Daily Log – {__import__('datetime').date.today().isoformat()}":
            print("No meaningful daily log to reflect on.")
            return

        print("Running nightly reflection with Claude Opus...")
        result = analyst.reflect_on_day(daily_log, learnings)

        # Update learnings.md
        if result.new_learnings:
            log.update_learnings(result.new_learnings)
            print(f"Added {len(result.new_learnings)} new learnings.")

        # Update watchlist notes
        if result.watchlist_updates:
            log.update_watchlist(result.watchlist_updates)
            print(f"Updated {len(result.watchlist_updates)} watchlist entries.")

        # Append reflection summary to daily log
        summary_lines = [f"**{len(result.new_learnings)} neue Lernerkenntnisse**"]
        if result.benchmark_comment:
            summary_lines.append(f"**MSCI World Vergleich:** {result.benchmark_comment}")
        if result.tomorrow_focus:
            summary_lines.append(f"**Morgen im Fokus:** {result.tomorrow_focus}")
        if result.risk_flags:
            summary_lines.append(f"**Risikohinweise:** {', '.join(result.risk_flags)}")
        if result.new_learnings:
            summary_lines.append("\n**Neue Erkenntnisse:**")
            for lesson in result.new_learnings:
                summary_lines.append(f"- {lesson}")
        log.append_to_daily_log("Nightly Reflection", "\n".join(summary_lines))

        # Telegram confirmation
        telegram.send_reflection_complete(len(result.new_learnings))

        print("Nightly reflection complete.")

    except Exception as e:
        print(f"Nightly reflection error: {e}\n{traceback.format_exc()}")
        try:
            telegram.send_error("nightly_reflection", str(e))
        except Exception:
            pass
        sys.exit(1)
