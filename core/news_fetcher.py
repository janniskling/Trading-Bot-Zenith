from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

import feedparser
import yfinance as yf


RSS_FEEDS = {
    "reuters": "https://feeds.reuters.com/reuters/businessNews",
    "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
    "marketwatch": "http://feeds.marketwatch.com/marketwatch/topstories/",
    "seeking_alpha": "https://seekingalpha.com/market_currents.xml",
}


@dataclass
class NewsItem:
    symbol: str
    headline: str
    summary: str
    source: str
    published: str


@dataclass
class EarningsEvent:
    symbol: str
    earnings_date: date
    days_away: int


def fetch_rss_headlines(symbols: list[str], max_per_symbol: int = 3) -> list[NewsItem]:
    items: list[NewsItem] = []
    for source_name, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:30]:
                title = entry.get("title", "")
                summary = entry.get("summary", "")[:300]
                published = entry.get("published", "")
                text = (title + " " + summary).upper()
                for sym in symbols:
                    if sym in text:
                        items.append(NewsItem(
                            symbol=sym,
                            headline=title,
                            summary=summary,
                            source=source_name,
                            published=published,
                        ))
                        break
        except Exception:
            pass
    return items


def fetch_earnings_calendar(symbols: list[str]) -> list[EarningsEvent]:
    today = date.today()
    events: list[EarningsEvent] = []
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            cal = ticker.calendar
            if cal is None:
                continue
            # calendar returns a dict with 'Earnings Date' key
            if hasattr(cal, "columns") and "Earnings Date" in cal.columns:
                for ed in cal["Earnings Date"]:
                    if hasattr(ed, "date"):
                        ed = ed.date()
                    if isinstance(ed, date) and ed >= today:
                        days_away = (ed - today).days
                        events.append(EarningsEvent(symbol=sym, earnings_date=ed, days_away=days_away))
        except Exception:
            pass
    return events


def fetch_market_movers() -> dict:
    """Returns pre-market gainers and losers from yfinance."""
    try:
        # Use common index ETFs as a proxy for market direction
        spy = yf.Ticker("SPY").fast_info
        qqq = yf.Ticker("QQQ").fast_info
        return {
            "spy_change_pct": round((spy.last_price / spy.previous_close - 1) * 100, 2),
            "qqq_change_pct": round((qqq.last_price / qqq.previous_close - 1) * 100, 2),
        }
    except Exception:
        return {}


def derive_market_mood(movers: dict) -> str:
    spy_pct = movers.get("spy_change_pct", 0)
    qqq_pct = movers.get("qqq_change_pct", 0)
    avg = (spy_pct + qqq_pct) / 2
    if avg >= 0.5:
        return f"Bullish (SPY {spy_pct:+.1f}%, QQQ {qqq_pct:+.1f}%)"
    elif avg <= -0.5:
        return f"Bearish (SPY {spy_pct:+.1f}%, QQQ {qqq_pct:+.1f}%)"
    else:
        return f"Neutral (SPY {spy_pct:+.1f}%, QQQ {qqq_pct:+.1f}%)"


def format_news_for_prompt(news_items: list[NewsItem], earnings: list[EarningsEvent]) -> str:
    if not news_items and not earnings:
        return "Keine relevanten News gefunden."

    lines: list[str] = []
    by_symbol: dict[str, list[NewsItem]] = {}
    for item in news_items:
        by_symbol.setdefault(item.symbol, []).append(item)

    for sym, items in by_symbol.items():
        lines.append(f"**{sym}:**")
        for it in items[:3]:
            lines.append(f"  - [{it.source}] {it.headline}")

    if earnings:
        lines.append("\n**Upcoming Earnings (avoid entry):**")
        for e in earnings:
            if e.days_away <= 3:
                lines.append(f"  - {e.symbol}: {e.earnings_date} ({e.days_away}d away) ⚠️")

    return "\n".join(lines)
