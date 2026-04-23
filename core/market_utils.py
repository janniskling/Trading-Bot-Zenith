from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

BERLIN = ZoneInfo("Europe/Berlin")
UTC = timezone.utc


def should_skip_today(alpaca_client=None) -> bool:
    """Returns True on weekends or US market holidays. Uses Alpaca clock as source of truth."""
    if alpaca_client is not None:
        try:
            clock = alpaca_client.get_clock()
            # If market is currently open, definitely don't skip
            if clock["is_open"]:
                return False
            # Check if there's a next_open today (i.e. pre-market, not a holiday)
            next_open = clock["next_open"]
            if hasattr(next_open, "date"):
                next_open_date = next_open.date()
            else:
                next_open_date = datetime.fromisoformat(str(next_open)).date()

            today = date.today()
            if next_open_date > today:
                # Next open is tomorrow or later — market is not trading today
                return True
            return False
        except Exception:
            pass

    # Fallback: skip weekends
    today = date.today()
    return today.weekday() >= 5  # 5=Saturday, 6=Sunday


def is_market_open(alpaca_client) -> bool:
    clock = alpaca_client.get_clock()
    return clock["is_open"]


def format_german_time(dt: datetime) -> str:
    """Converts a UTC-aware datetime to a Berlin-local formatted string."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    berlin_dt = dt.astimezone(BERLIN)
    return berlin_dt.strftime("%d.%m.%Y %H:%M Uhr")


def format_utc_and_berlin(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    berlin_dt = dt.astimezone(BERLIN)
    return f"{dt.strftime('%H:%M')} UTC / {berlin_dt.strftime('%H:%M')} Berlin"


def today_iso() -> str:
    return date.today().isoformat()


def count_trading_days_since(start_date_iso: str) -> int:
    """Approximate count of weekday trading days since start date."""
    if not start_date_iso:
        return 0
    try:
        start = date.fromisoformat(start_date_iso)
        today = date.today()
        if today <= start:
            return 0
        count = 0
        current = start
        while current < today:
            if current.weekday() < 5:
                count += 1
            current = date(current.year, current.month, current.day + 1) \
                if current.day < 28 else _next_day(current)
        return count
    except Exception:
        return 0


def _next_day(d: date) -> date:
    from datetime import timedelta
    return d + timedelta(days=1)
