"""Current time / timezone helpers."""
from __future__ import annotations

import datetime
import zoneinfo


def get_current_time(timezone: str = "UTC") -> str:
    try:
        tz = zoneinfo.ZoneInfo(timezone)
    except (KeyError, zoneinfo.ZoneInfoNotFoundError):
        return f"Unknown timezone: {timezone!r}. Use an IANA name like 'America/New_York'."
    now = datetime.datetime.now(tz)
    return now.strftime("%A, %B %d %Y %I:%M %p %Z")
