from __future__ import annotations

import re
import shutil
from datetime import date, datetime, timezone
from pathlib import Path

from config.settings import DAILY_LOG_PATH, LEARNINGS_PATH, LOGS_DIR, WATCHLIST_PATH

MAX_LEARNINGS = 200


class LogManager:
    # ----- Daily Log -----

    def init_daily_log(self, today: str | None = None) -> None:
        today = today or date.today().isoformat()
        header = f"# Daily Log – {today}\n\n"
        DAILY_LOG_PATH.write_text(header, encoding="utf-8")

    def append_to_daily_log(self, section: str, content: str) -> None:
        now = datetime.now(tz=timezone.utc).strftime("%H:%M UTC")
        block = f"\n## {section}\n*{now}*\n\n{content}\n"
        with DAILY_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(block)

    def read_daily_log(self) -> str:
        if not DAILY_LOG_PATH.exists():
            return ""
        return DAILY_LOG_PATH.read_text(encoding="utf-8")

    def archive_daily_log(self) -> None:
        if not DAILY_LOG_PATH.exists():
            return
        today = date.today().isoformat()
        archive_path = LOGS_DIR / f"{today}.md"
        shutil.copy2(DAILY_LOG_PATH, archive_path)

    # ----- Learnings -----

    def read_learnings(self) -> str:
        if not LEARNINGS_PATH.exists():
            return ""
        return LEARNINGS_PATH.read_text(encoding="utf-8")

    def update_learnings(self, new_learnings: list[str]) -> None:
        if not new_learnings:
            return
        today = date.today().isoformat()
        entries = "\n".join(f"- {lesson}" for lesson in new_learnings)
        new_block = f"\n## {today}\n{entries}\n"

        existing = self.read_learnings()
        if not existing:
            content = f"# Zenith Trading Learnings\n{new_block}"
        else:
            # Prepend after the header line
            lines = existing.split("\n", 2)
            if len(lines) >= 2:
                content = lines[0] + "\n" + new_block + "\n".join(lines[1:])
            else:
                content = existing + new_block

        # Trim to MAX_LEARNINGS bullet points
        content = self._trim_learnings(content)
        LEARNINGS_PATH.write_text(content, encoding="utf-8")

    def _trim_learnings(self, content: str) -> str:
        bullets = [line for line in content.split("\n") if line.startswith("- ")]
        if len(bullets) <= MAX_LEARNINGS:
            return content
        # Keep only the last MAX_LEARNINGS bullets by rebuilding from scratch is complex
        # Simple approach: truncate trailing old entries
        lines = content.split("\n")
        bullet_count = 0
        cut_index = len(lines)
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith("- "):
                bullet_count += 1
                if bullet_count > MAX_LEARNINGS:
                    cut_index = i
        return "\n".join(lines[:cut_index])

    # ----- Watchlist -----

    def read_watchlist(self) -> tuple[str, list[str]]:
        if not WATCHLIST_PATH.exists():
            return "", []
        content = WATCHLIST_PATH.read_text(encoding="utf-8")
        # Extract ticker symbols: lines starting with "### TICKER" or "- $TICKER"
        symbols: list[str] = []
        for line in content.split("\n"):
            m = re.match(r"^###\s+([A-Z]{1,5})\b", line)
            if m:
                symbols.append(m.group(1))
            m2 = re.match(r"^-\s+\$([A-Z]{1,5})\b", line)
            if m2:
                symbols.append(m2.group(1))
        return content, list(dict.fromkeys(symbols))  # deduplicated, order preserved

    def add_symbol_to_watchlist(self, symbol: str, thesis: str, segment: str = "Discovered") -> bool:
        """Adds a new symbol to the watchlist. Returns False if already present."""
        content, existing_symbols = self.read_watchlist()
        if symbol.upper() in existing_symbols:
            return False
        today = date.today().isoformat()
        new_entry = (
            f"\n## {segment}\n\n"
            f"### {symbol.upper()}\n"
            f"**Segment:** {segment}  \n"
            f"**Grund für Beobachtung:** {thesis}  \n"
            f"**Strategie:** EMA-Crossover auf Daily Chart  \n"
            f"**Bot notes**: (hinzugefügt {today})\n"
        )
        WATCHLIST_PATH.write_text(content + new_entry, encoding="utf-8")
        return True

    def update_watchlist(self, notes: dict[str, str]) -> None:
        """Updates **Bot notes** section for tickers in the watchlist."""
        if not notes:
            return
        content = self.read_watchlist()[0]
        today = date.today().isoformat()
        for symbol, note in notes.items():
            # Replace existing bot note line or append one
            pattern = rf"(\*\*Bot notes.*?\*\*:).*"
            replacement = f"\\1 ({today}) {note}"
            new_content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
            if new_content == content:
                # Symbol section exists but no bot note line — find section and add
                section_pattern = rf"(### {symbol}.*?\n)"
                content = re.sub(
                    section_pattern,
                    f"\\1**Bot notes**: ({today}) {note}\n",
                    content,
                    flags=re.DOTALL,
                )
            else:
                content = new_content
        WATCHLIST_PATH.write_text(content, encoding="utf-8")

    # ----- Parsing helpers -----

    def parse_candidates_from_log(self, log_content: str) -> list[str]:
        """Extract ticker symbols from Pre-Market section of daily log."""
        symbols: list[str] = []
        in_premarket = False
        for line in log_content.split("\n"):
            if "## Pre-Market" in line:
                in_premarket = True
            elif line.startswith("## ") and in_premarket:
                break
            if in_premarket:
                m = re.findall(r"\b([A-Z]{1,5})\b", line)
                symbols.extend(m)
        # Filter known non-tickers (includes single-letter words common in prose)
        skip = {"BUY", "SELL", "RSI", "EMA", "SL", "TP", "UTC", "ETF", "MA",
                "A", "I", "S", "Q", "P", "N", "M", "K", "J", "H", "G", "F",
                "E", "D", "C", "B", "OR", "IN", "OF", "TO", "AT", "VS", "AI"}
        return [s for s in dict.fromkeys(symbols) if s not in skip]
